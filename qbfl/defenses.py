"""Aggregation rules and contract-enforced acceptance rules.

A Defense sees the matrix of submitted updates U [M, P] (after reveal) and
returns per-update weights. Contract-style defenses also keep state across
rounds (reputation, bonds, burned probes).

What an attacker may read before it submits is exposed by `public_view()`.
For public rules this is the whole rule. For the sealed rule it is only the
commitments and the declared check types, not the probe samples or the
dynamic thresholds.

Baselines are our own re-implementations from the descriptions in the papers.
They are named "-style" in the paper because the original code was not used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

from . import model as M


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a @ b) / (a.norm(dim=-1) * b.norm() + 1e-12) if a.dim() == 2 else (a @ b) / (a.norm() * b.norm() + 1e-12)


@dataclass
class Ctx:
    """Round context passed to every defense."""
    round: int
    theta: torch.Tensor          # current global model [P]
    shapes: list
    n_clients: int
    beacon: bytes = b""
    sizes: torch.Tensor | None = None


class Defense:
    name = "base"
    uses_state = False

    def __init__(self, n_clients: int, device):
        self.n = n_clients
        self.device = device
        self.rep = torch.ones(n_clients, device=device)
        self.banned = torch.zeros(n_clients, dtype=torch.bool, device=device)
        self.bond = torch.ones(n_clients, device=device)
        self.log = {}
        self.last_w = None

    # ---- interface ----
    def weights(self, U: torch.Tensor, ctx: Ctx) -> torch.Tensor:  # [M] non-negative
        raise NotImplementedError

    def aggregate(self, U: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        w = self.weights(U, ctx)
        w = w * (~self.banned).float()
        self.last_w = w.detach()
        if w.sum() <= 0:
            return torch.zeros(U.shape[1], device=U.device)
        return (w / w.sum()) @ U

    def public_view(self) -> dict:
        return {}

    def accepts(self, u: torch.Tensor, ref: torch.Tensor, ctx: Ctx, client: int) -> bool:
        """Attacker-side dry run of the PUBLIC part of the rule. `ref` is the
        attacker's own estimate of the benign mean update."""
        return True


# ------------------------------------------------------------------ plain aggregators
class FedAvg(Defense):
    name = "fedavg"

    def weights(self, U, ctx):
        return torch.ones(U.shape[0], device=U.device)


class CoordMedian(Defense):
    name = "median"

    def aggregate(self, U, ctx):
        self.last_w = None
        return U.median(0).values


class TrimmedMean(Defense):
    name = "trmean"

    def __init__(self, n_clients, device, beta=0.2):
        super().__init__(n_clients, device)
        self.beta = beta

    def aggregate(self, U, ctx):
        self.last_w = None
        k = int(self.beta * U.shape[0])
        S = U.sort(0).values
        return S[k:U.shape[0] - k].mean(0) if k > 0 else S.mean(0)


class MultiKrum(Defense):
    name = "mkrum"

    def __init__(self, n_clients, device, f_frac=0.3, m=None):
        super().__init__(n_clients, device)
        self.f_frac = f_frac
        self.f = None
        self.m = m

    def weights(self, U, ctx):
        n = U.shape[0]
        f = max(1, int(self.f_frac * n))
        k = max(1, n - f - 2)
        D = torch.cdist(U, U) ** 2
        sc = torch.stack([D[i][torch.arange(n, device=U.device) != i].sort().values[:k].sum() for i in range(n)])
        m = self.m or max(1, n - f)
        w = torch.zeros(n, device=U.device)
        w[sc.argsort()[:m]] = 1
        return w


# ------------------------------------------------------------------ contract rules (public)
class NormGate(Defense):
    """Norm-threshold contract (Nagarjun & Rajkumar style): reject if the update norm exceeds tau.
    tau is public: `tau_mult` x the running median honest norm, published each round."""
    name = "normgate"
    uses_state = True

    def __init__(self, n_clients, device, tau_mult=1.5):
        super().__init__(n_clients, device)
        self.tau_mult = tau_mult
        self.tau = float("inf")
        self.ref_norm = None

    def weights(self, U, ctx):
        norms = U.norm(dim=1)
        med = norms.median().item()
        self.ref_norm = med if self.ref_norm is None else 0.9 * self.ref_norm + 0.1 * med
        self.tau = self.tau_mult * self.ref_norm
        w = (norms <= self.tau).float()
        # reference norm uses only accepted updates next round (public state)
        if w.sum() > 0:
            self.ref_norm = float(norms[w > 0].median())
        return w

    def public_view(self):
        return {"tau": self.tau, "ref_norm": self.ref_norm}

    def accepts(self, u, ref, ctx, client):
        tau = self.tau_mult * (self.ref_norm if self.ref_norm is not None else ref.norm().item())
        return bool(u.norm() <= tau)


class CosRep(Defense):
    """PFAD-BC-style: reputation-weighted cosine aggregation (reconstructed from the abstract).
    Reference direction = coordinate median. Reputation is an EMA of clipped cosine; tenants
    below rho are dropped. Norms are clipped to the median norm."""
    name = "cosrep"
    uses_state = True

    def __init__(self, n_clients, device, lam=0.8, rho=0.3, cos_min=0.0):
        super().__init__(n_clients, device)
        self.lam, self.rho, self.cos_min = lam, rho, cos_min
        self.rep = torch.full((n_clients,), 0.5, device=device)

    def weights(self, U, ctx):
        ref = U.median(0).values
        cs = torch.clamp(_cos(U, ref), min=0)
        self.rep = self.lam * self.rep + (1 - self.lam) * cs
        w = self.rep * (self.rep >= self.rho).float() * (cs > self.cos_min).float()
        norms = U.norm(dim=1)
        clip = torch.clamp(norms.median() / (norms + 1e-12), max=1.0)
        self._clip = clip
        return w * clip

    def public_view(self):
        return {"rep": self.rep.clone(), "rho": self.rho, "lam": self.lam}

    def accepts(self, u, ref, ctx, client):
        # public rule: cosine to the benign direction must stay > cos_min and reputation >= rho
        c = float(_cos(u, ref))
        return c > max(self.cos_min, 0.05)


class TrustAware(Defense):
    """Trust-aware aggregation (Houichi et al. style): per-round weights from distance to the
    weighted mean, smoothed into a trust score across rounds."""
    name = "trustaware"
    uses_state = True

    def __init__(self, n_clients, device, lam=0.7, temp=2.0):
        super().__init__(n_clients, device)
        self.lam, self.temp = lam, temp
        self.trust = torch.full((n_clients,), 0.5, device=device)

    def weights(self, U, ctx):
        mu = (self.trust / self.trust.sum()) @ U
        d = (U - mu).norm(dim=1)
        s = torch.exp(-self.temp * d / (d.median() + 1e-12))
        self.trust = self.lam * self.trust + (1 - self.lam) * s
        return self.trust * (self.trust > 0.15).float()

    def public_view(self):
        return {"trust": self.trust.clone(), "lam": self.lam}

    def accepts(self, u, ref, ctx, client):
        return bool((u - ref).norm() <= 2.0 * ref.norm())


# ------------------------------------------------------------------ probe-based rules
def probe_stats(theta_g: torch.Tensor, U: torch.Tensor, shapes, X: torch.Tensor, fam: np.ndarray, families):
    """Per-family attack recall and benign false-alarm rate of theta_g and of every theta_g + u_i on a probe batch.

    Returns R [C+1, F] (row 0 = current global model), fpr [C+1]."""
    cand = torch.cat([theta_g.unsqueeze(0), theta_g.unsqueeze(0) + U], 0)
    pred = M.predict(cand, shapes, X)
    dev = X.device
    ben = torch.as_tensor(fam == "benign", device=dev)
    fpr = (pred[:, ben] == 1).float().mean(1) if ben.any() else torch.zeros(cand.shape[0], device=dev)
    cols = []
    for f in families:
        if f == "benign":
            continue
        m = torch.as_tensor(fam == f, device=dev)
        cols.append((pred[:, m] == 1).float().mean(1) if m.any() else torch.ones(cand.shape[0], device=dev))
    return torch.stack(cols, 1), fpr


def damage(theta_g: torch.Tensor, U: torch.Tensor, shapes, X: torch.Tensor, fam: np.ndarray, families):
    """Worst-case harm of applying each update to the global model on a probe batch.

    harm = max( largest drop in any attack-family recall , rise in benign false-alarm rate ).
    theta_g [P], U [C,P], X [N,in], fam [N] -> harm [C]
    """
    R, fpr = probe_stats(theta_g, U, shapes, X, fam, families)
    drop = (R[0:1] - R[1:]).max(1).values
    return torch.maximum(drop, fpr[1:] - fpr[0]), None


def peer_z(R: torch.Tensor, fpr: torch.Tensor, s_min: float, ref: tuple | None = None):
    """Peer-relative badness. Each column is one check (attack-family recall, benign false alarms);
    an update is judged against the median and MAD of the OTHER submitted updates in that column.

    R [M,F] recall, fpr [M]. Returns z [M,F+1]. `ref=(med, mad)` lets an attacker plug in its own estimate."""
    S = torch.cat([-R, fpr.unsqueeze(1)], 1)              # larger = worse
    if ref is None:
        med = S.median(0).values
        mad = (S - med).abs().median(0).values
    else:
        med, mad = ref
    return (S - med) / torch.clamp(mad, min=s_min), (med, mad)


class ProbeRule(Defense):
    """Accept an update only if it does not damage detection on a probe batch.

    Two independent design choices, kept separate so the experiments can tell which one matters:

    public (probe source)
        True : one fixed probe set, readable by everyone (FLTrust-style root data on chain).
        False: SEALED. The pool is committed by Merkle root at setup. Each round a beacon picks fresh
               samples that were never used before ("burn"). They are revealed only after the
               submissions close, so nobody can fit to them in advance.
    mode (pass mark)
        "abs"    : fixed tolerance learned from clean calibration rounds and published on chain.
        "rel_max": the worst-case harm of an update must stay within median + kappa*MAD of the harm of
                   the other updates (unknown before reveal).
        "rel_fam": per-check peer test. Each family recall and the benign false-alarm rate is compared
                   with the median and MAD of the other updates for that same check; reject when any
                   check is more than kappa robust deviations worse (unknown before reveal).
    """
    uses_state = True

    def __init__(self, n_clients, device, pool, families, public=False, mode="rel_max", probe_size=1024,
                 kappa=3.0, delta_floor=0.02, s_min=0.05, public_tol=0.15, calib_rounds=4, calib_q=0.95,
                 rep_lam=0.8, slash_margin=0.25, reuse=False, leak_frac=0.0, seed=0, use_rep=True):
        super().__init__(n_clients, device)
        self.public, self.mode = public, mode
        self.name = {("abs", True): "probe_public", ("abs", False): "sealed_abs", ("rel_max", False): "sealed",
                     ("rel_max", True): "public_rel", ("rel_fam", True): "public_fam", ("rel_fam", False): "sealed_fam"}[(mode, public)]
        self.pool_X = torch.as_tensor(pool["X"], device=device)
        self.pool_fam = np.asarray(pool["fam"])
        self.families = [f for f in families if f != "benign"]
        self.probe_size = probe_size
        self.kappa, self.delta_floor, self.s_min, self.public_tol = kappa, delta_floor, s_min, public_tol
        self.public_thr = delta_floor + public_tol   # the only absolute pass mark an attacker can read
        self.calib_rounds, self.calib_q, self._calib = calib_rounds, calib_q, []
        self.rep_lam, self.slash_margin = rep_lam, slash_margin
        self.reuse, self.leak_frac, self.use_rep = reuse, leak_frac, use_rep
        self.rng = np.random.default_rng(seed)
        self.used = np.zeros(len(self.pool_fam), dtype=bool)
        self.rep = torch.full((n_clients,), 0.7, device=device)
        self.root = hashlib.sha256(np.ascontiguousarray(pool["X"][:64]).tobytes() + str(len(pool["fam"])).encode()).hexdigest()
        self.public_idx = self._draw(probe_size, burn=False) if public else None
        self.slashed = torch.zeros(n_clients, device=device)
        self.last = {}

    # ---- probe drawing
    def _draw(self, n, burn=True, seed_bytes=b""):
        avail = np.flatnonzero(~self.used) if (burn and not self.reuse) else np.arange(len(self.pool_fam))
        if len(avail) < n:  # pool exhausted: refresh (documented as a limitation)
            self.used[:] = False
            avail = np.arange(len(self.pool_fam))
        r = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_bytes + str(self.rng.integers(1 << 30)).encode()).digest()[:8], "little"))
        fams = np.unique(self.pool_fam[avail])           # stratify so every family is checked every round
        per = max(1, n // len(fams))
        sel = []
        for f in fams:
            cand = avail[self.pool_fam[avail] == f]
            sel.append(r.choice(cand, min(per, len(cand)), replace=False))
        sel = np.concatenate(sel)
        if burn and not self.reuse:
            self.used[sel] = True
        return sel

    def _probe(self, ctx):
        idx = self.public_idx if self.public else self._draw(self.probe_size, burn=True, seed_bytes=ctx.beacon)
        self.last_idx = idx
        return idx

    def weights(self, U, ctx):
        idx = self._probe(ctx)
        Xp, famp = self.pool_X[idx], self.pool_fam[idx]
        R, fpr = probe_stats(ctx.theta, U, ctx.shapes, Xp, famp, self.families)
        if self.mode == "rel_fam":
            z, _ = peer_z(R[1:], fpr[1:], self.s_min)
            dmg = z.max(1).values
            thr = torch.as_tensor(self.kappa, device=U.device)
        else:
            dmg = torch.maximum((R[0:1] - R[1:]).max(1).values, fpr[1:] - fpr[0])
            if self.mode == "abs":
                if ctx.round < self.calib_rounds:   # clean calibration rounds: accept all, learn the honest spread
                    self._calib.append(dmg.detach().cpu().numpy())
                    if ctx.round == self.calib_rounds - 1:
                        self.public_thr = float(np.quantile(np.concatenate(self._calib), self.calib_q)) + self.delta_floor
                    thr = torch.as_tensor(float("inf"), device=U.device)
                else:
                    thr = torch.as_tensor(self.public_thr, device=U.device)
            else:                                    # rel_max
                med = dmg.median()
                mad = (dmg - med).abs().median()
                thr = med + self.kappa * (mad + 1e-3) + self.delta_floor
        ok = (dmg <= thr).float()
        self.last = dict(dmg=dmg.detach().cpu().numpy(), thr=float(thr), ok=ok.detach().cpu().numpy())
        scale = float(thr) if np.isfinite(float(thr)) and float(thr) > 1e-6 else 1.0
        if self.use_rep:
            score = 1 - torch.clamp(dmg / scale, 0, 2) / 2
            self.rep = self.rep_lam * self.rep + (1 - self.rep_lam) * score
        # slash with attributable evidence (signed commitment + revealed update + revealed probes)
        margin = self.slash_margin * (scale if self.mode == "rel_fam" else 1.0)
        bad = dmg > (thr + margin)
        self.slashed += bad.float() * 0.5
        self.banned |= self.slashed >= 1.0
        return ok * (torch.clamp(self.rep, min=0.05) if self.use_rep else 1.0)

    def public_view(self):
        v = {"root": self.root, "probe_size": self.probe_size, "delta_floor": self.delta_floor,
             "kappa": self.kappa, "s_min": self.s_min, "mode": self.mode, "rep": self.rep.clone()}
        if self.public:
            v["probe_idx"] = self.public_idx
            v["pool_X"], v["pool_fam"] = self.pool_X, self.pool_fam
        elif self.leak_frac > 0:
            n = int(self.leak_frac * len(self.pool_fam))
            leak = np.random.default_rng(123).choice(len(self.pool_fam), n, replace=False)
            v["leak_idx"] = leak
            v["pool_X"], v["pool_fam"] = self.pool_X, self.pool_fam
        return v

    def accepts(self, u, ref, ctx, client):
        return True  # handled in the attacker via the probe it can actually see


PROBE_NAMES = {"probe_public": (True, "abs"), "sealed_abs": (False, "abs"), "sealed": (False, "rel_max"),
               "public_rel": (True, "rel_max"), "public_fam": (True, "rel_fam"), "sealed_fam": (False, "rel_fam")}


def make_defense(name: str, n_clients: int, device, audit_pool=None, families=None, seed=0, **kw) -> Defense:
    if name == "fedavg":
        return FedAvg(n_clients, device)
    if name == "median":
        return CoordMedian(n_clients, device)
    if name == "trmean":
        return TrimmedMean(n_clients, device, **kw)
    if name == "mkrum":
        return MultiKrum(n_clients, device, **kw)
    if name == "normgate":
        return NormGate(n_clients, device, **kw)
    if name == "cosrep":
        return CosRep(n_clients, device, **kw)
    if name == "trustaware":
        return TrustAware(n_clients, device, **kw)
    if name in PROBE_NAMES:
        public, mode = PROBE_NAMES[name]
        return ProbeRule(n_clients, device, audit_pool, families, public=public, mode=mode, seed=seed, **kw)
    raise KeyError(name)
