"""Attackers.

Goal of every attacker: make the global model miss a target attack family
(default "dos") while overall accuracy stays high. This is a stealthy
evasion goal; it is what a real intruder wants from a shared IDS.

Fixed attackers (do not read the contract):
    targeted_lf  label-flip data poisoning
    signflip     sign-flipped honest update
    lie          "a little is enough" (Baruch et al. 2019)
    minmax       AGR-agnostic min-max (Shejwalkar & Houmansadr 2021)

Rule-aware attacker (the study of this paper). It reads the PUBLIC part of the
acceptance rule from the chain and picks, by bisection, the largest push toward
its goal that the rule still accepts. It may also farm reputation first.
Attackers see only what a tenant could see: their own data, their own honest
updates, the public rule state and (if the contract has no commit-reveal) the
current round's benign updates.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from . import defenses as D


class Attacker:
    def __init__(self, kind: str, mal_idx, n_clients: int, farm_rounds: int = 0, gamma_max: float = 4.0,
                 see_benign: bool = False, margin: float = 0.9, surrogate_n: int = 600, seed: int = 0):
        self.kind = kind
        self.mal = list(mal_idx)
        self.n = n_clients
        self.farm = farm_rounds
        self.gamma_max = gamma_max
        self.see_benign = see_benign
        self.margin = margin
        self.surrogate_n = surrogate_n
        self.rng = np.random.default_rng(seed)
        self.last_gamma = 0.0
        self.train = None          # dict(epochs, bs, lr, gen, device) set by run_one
        self.target = "dos"
        self.leak_cap = 8000       # how many leaked pool samples a memorizing attacker can use
        self._surr = None          # persistent stand-in copy of the defense (opt_adaptive)
        self._prev_a = None
        self.make_surrogate = None # factory(pool) -> a fresh defense of the same kind, set by run_one
        self.families = None

    # --------------------------------------------------------------- helpers
    def _proxies(self, honest_U, mal_idx_t, benign_U=None):
        if self.see_benign and benign_U is not None:
            return benign_U
        return honest_U[mal_idx_t]

    def _surrogate(self, tenants, defense, ctx):
        """Attacker's stand-in for the validation probe: stratified samples of its own clean data
        (plus any leaked pool samples). Used only against sealed rules."""
        Xs, fs = [], []
        for i in self.mal:
            t = tenants[i]
            Xs.append(t["X"]); fs.append(t["fam"])
        X = np.concatenate(Xs); fam = np.concatenate(fs)
        sel = []
        for f in np.unique(fam):
            ii = np.flatnonzero(fam == f)
            sel.append(self.rng.choice(ii, min(len(ii), max(1, self.surrogate_n // len(np.unique(fam)))), replace=False))
        sel = np.concatenate(sel)
        Xp, fp = X[sel], fam[sel]
        v = defense.public_view()
        if "leak_idx" in v:
            li = v["leak_idx"][: max(self.surrogate_n, self.leak_cap if self.kind == "probe_memorize" else 0)]
            Xp = np.concatenate([Xp, v["pool_X"][li].cpu().numpy()])
            fp = np.concatenate([fp, v["pool_fam"][li]])
        return torch.as_tensor(Xp, device=ctx.theta.device), fp

    # --------------------------------------------------------------- main
    @torch.no_grad()
    def craft(self, honest_U: torch.Tensor, poison_U: torch.Tensor, ctx, defense, tenants, benign_U=None):
        """honest_U [C,P] honest updates of all clients; poison_U [m,P] updates trained on flipped
        target labels (rows follow self.mal). Returns submitted updates for the malicious clients [m,P]."""
        mi = torch.as_tensor(self.mal, device=honest_U.device)
        hu = honest_U[mi]
        k = self.kind
        if k == "none":
            return hu
        if k == "targeted_lf":
            return poison_U
        if k == "signflip":
            return -hu
        if k == "lie":
            mu, sd = hu.mean(0), hu.std(0) + 1e-12
            n, m = self.n, len(self.mal)
            s = int(np.floor(n / 2 + 1) - m)
            z = float(norm.ppf(min(max((n - s) / n, 0.5001), 0.9999)))
            return (mu - z * sd).unsqueeze(0).repeat(m, 1)
        if k == "minmax":
            prox = self._proxies(honest_U, mi, benign_U)
            mu = prox.mean(0)
            pdir = -mu / (mu.norm() + 1e-12)
            dmax = torch.cdist(prox, prox).max()
            lo, hi = 0.0, 50.0 * (mu.norm().item() + 1e-9)
            for _ in range(20):
                g = (lo + hi) / 2
                u = mu + g * pdir
                if (prox - u).norm(dim=1).max() <= dmax:
                    lo = g
                else:
                    hi = g
            self.last_gamma = lo
            return (mu + lo * pdir).unsqueeze(0).repeat(len(self.mal), 1)
        if k in ("opt_adaptive", "opt_adaptive_x2", "opt_adaptive_x4", "opt_adaptive_x8"):
            return self._opt_adaptive(honest_U, hu, poison_U, ctx, defense, tenants, mi, budget=int(k.split("_x")[1]) if "_x" in k else 1)
        if k == "opt_noisy":       # imperfect knowledge of honest updates: stand-ins 4x noisier
            return self._opt_adaptive(honest_U, hu, poison_U, ctx, defense, tenants, mi, noise=2.0)
        if k == "opt_lowdata":     # one tenth of the attacker's own flows for its objective and its stand-in probe
            return self._opt_adaptive(honest_U, hu, poison_U, ctx, defense, tenants, mi, eval_n=150, surr_n=400)
        if k == "opt_es":          # independent search: evolution strategy over a six-direction space, same budget
            return self._opt_es(honest_U, hu, poison_U, ctx, defense, tenants, mi)
        if k == "agr_tailored":    # Shejwalkar & Houmansadr (NDSS 2021) AGR-tailored attack against the (simulated) rule
            return self._agr_tailored(honest_U, hu, poison_U, ctx, defense, tenants, mi)
        if k == "probe_memorize":
            if ctx.round < 0 or defense.name not in D.PROBE_NAMES:
                return self._rule_aware(honest_U, hu, poison_U, ctx, defense, tenants, benign_U, mi)
            u_mem = self._memorize(hu, poison_U, ctx, defense, tenants)
            return self._rule_aware(honest_U, hu, u_mem, ctx, defense, tenants, benign_U, mi)
        if k in ("rule_aware", "rule_aware_farm"):
            if k == "rule_aware_farm" and ctx.round < self.farm:
                self.last_gamma = 0.0
                return hu                                    # build reputation first
            return self._rule_aware(honest_U, hu, poison_U, ctx, defense, tenants, benign_U, mi)
        raise KeyError(k)

    # --------------------------------------------------------------- defense-agnostic optimiser
    def _opt_adaptive(self, honest_U, hu, pu, ctx, defense, tenants, mi, budget: int = 1, noise: float = 0.5,
                      eval_n: int = 1500, surr_n: int = 4000):
        """End-to-end adaptive attack that works against any rule.

        The attacker keeps a private stand-in copy of the defense (same rule, same constants, same public state, but
        its OWN probe pool: the exact probe if the rule is public, a stand-in built from its own data otherwise).
        Each round it builds a stand-in population (its own honest updates, resampled with noise, as the benign
        tenants), then searches over a small family of update shapes
            u_i = h_i + a1 (p_i - h_i) + a2 (-h_i) + a3 (-unit(mean h) |h_i|)
        and keeps the one that does the most damage to the AGGREGATE produced by the stand-in defense, measured as
        the drop in DoS recall on its own DoS samples (with a small penalty for extra false alarms).
        No assumption about the rule is built in, so the same code attacks median, Krum, CosRep, norm gates and
        probe rules alike."""
        import copy
        from . import model as M
        dev, m, N = hu.device, len(self.mal), self.n
        rng = self.rng
        X_own = np.concatenate([tenants[i]["X"] for i in self.mal]); F_own = np.concatenate([tenants[i]["fam"] for i in self.mal])
        ti = np.flatnonzero(F_own == self.target); bi = np.flatnonzero(F_own == "benign")
        if len(ti) == 0 or len(bi) == 0:
            return hu
        ti = rng.choice(ti, min(len(ti), eval_n), replace=False); bi = rng.choice(bi, min(len(bi), eval_n), replace=False)
        Xt, Xb = torch.as_tensor(X_own[ti], device=dev), torch.as_tensor(X_own[bi], device=dev)

        if self._surr is None:                                   # build the private stand-in defense once
            pool = None
            if defense.name in D.PROBE_NAMES:
                v = defense.public_view()
                if defense.public:
                    idx = v["probe_idx"]
                    Kx, Kf = v["pool_X"][idx].cpu().numpy(), np.asarray(v["pool_fam"])[idx]
                else:
                    old = self.surrogate_n; self.surrogate_n = surr_n
                    Xp, Kf = self._surrogate(tenants, defense, ctx); self.surrogate_n = old
                    Kx, Kf = Xp.cpu().numpy(), np.asarray(Kf)
                pool = {"X": Kx, "y": (Kf != "benign").astype(np.int64), "fam": Kf}
            self._surr = self.make_surrogate(pool)
        surr = self._surr
        for a in ("rep", "trust", "public_thr", "ref_norm", "tau", "banned"):   # public on-chain state is readable
            if hasattr(defense, a) and hasattr(surr, a):
                val = getattr(defense, a)
                setattr(surr, a, val.clone() if torch.is_tensor(val) else val)

        honest_idx = [i for i in range(N) if i not in set(self.mal)]
        mu = hu.mean(0); sd = hu.std(0) if m > 1 else hu.abs() * 0.2
        sd = torch.clamp(sd, min=1e-8)
        g = torch.Generator(device=dev).manual_seed(int(rng.integers(1 << 30)))
        prox = hu[torch.randint(0, m, (len(honest_idx),), generator=g, device=dev)] + noise * sd * torch.randn(len(honest_idx), hu.shape[1], generator=g, device=dev)
        mdir = -mu / (mu.norm() + 1e-12)
        from .defenses import Ctx
        cs = Ctx(ctx.round, ctx.theta, ctx.shapes, N, ctx.beacon)

        def build(a):
            U = torch.zeros(N, hu.shape[1], device=dev)
            U[honest_idx] = prox
            U[self.mal] = hu + a[0] * (pu - hu) + a[1] * (-hu) + a[2] * mdir * hu.norm(dim=1, keepdim=True)
            return U

        def score(a):
            sc = copy.deepcopy(surr)
            delta = sc.aggregate(build(a), cs)
            th = ctx.theta + delta
            pt = M.predict(th, ctx.shapes, Xt)[0]; pb = M.predict(th, ctx.shapes, Xb)[0]
            return float((pt == 1).float().mean()), float((pb == 1).float().mean())

        rec0, fpr0 = score((0.0, 0.0, 0.0))
        cands = [(1, 0, 0), (2, 0, 0), (4, 0, 0), (8, 0, 0), (1, 1, 0), (2, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2), (1, 0, 1), (2, 0, 1)]
        if self._prev_a is not None:
            cands.append(tuple(self._prev_a))
        for _ in range(12 * budget):
            a = np.exp(rng.uniform(np.log(0.05), np.log(8.0), size=3)) * (rng.random(3) < 0.6)
            cands.append(tuple(float(x) for x in a))
        best, bj = (0.0, 0.0, 0.0), 0.0

        def consider(a):
            nonlocal best, bj
            rec, fpr = score(a)
            j = (rec0 - rec) - 0.5 * max(0.0, fpr - fpr0 - 0.02)
            if j > bj:
                best, bj = a, j
        for a in cands:
            consider(a)
        if budget > 1:                                           # local refinement around the incumbent
            for _ in range(12 * (budget - 1)):
                a = tuple(float(max(0.0, x * np.exp(rng.normal(0, 0.35)) + (0.05 * rng.random() if x == 0 and rng.random() < 0.2 else 0.0))) for x in best)
                consider(a)
        self._prev_a = best
        self.last_gamma = float(best[0])
        U_best = build(best)
        surr.aggregate(U_best, cs)                              # advance the stand-in's own state
        return U_best[self.mal]

    # --------------------------------------------------------------- independent attackers (R3)
    def _sim_setup(self, hu, ctx, defense, tenants, eval_n=1500, surr_n=4000, noise=0.5):
        """Shared set-up for the R3 attackers: own evaluation flows, private copy of the published rule (with public state
        copied from chain), stand-ins for the honest tenants. Mirrors _opt_adaptive's set-up; kept separate so that the
        pre-registered optimiser path stays byte-for-byte unchanged."""
        import copy
        from . import model as M
        from .defenses import Ctx
        dev, m, N = hu.device, len(self.mal), self.n
        rng = self.rng
        X_own = np.concatenate([tenants[i]["X"] for i in self.mal]); F_own = np.concatenate([tenants[i]["fam"] for i in self.mal])
        ti = np.flatnonzero(F_own == self.target); bi = np.flatnonzero(F_own == "benign")
        if len(ti) == 0 or len(bi) == 0:
            return None
        ti = rng.choice(ti, min(len(ti), eval_n), replace=False); bi = rng.choice(bi, min(len(bi), eval_n), replace=False)
        Xt, Xb = torch.as_tensor(X_own[ti], device=dev), torch.as_tensor(X_own[bi], device=dev)
        if self._surr is None:
            pool = None
            if defense.name in D.PROBE_NAMES:
                v = defense.public_view()
                if defense.public:
                    idx = v["probe_idx"]; Kx, Kf = v["pool_X"][idx].cpu().numpy(), np.asarray(v["pool_fam"])[idx]
                else:
                    old = self.surrogate_n; self.surrogate_n = surr_n
                    Xp, Kf = self._surrogate(tenants, defense, ctx); self.surrogate_n = old
                    Kx, Kf = Xp.cpu().numpy(), np.asarray(Kf)
                pool = {"X": Kx, "y": (Kf != "benign").astype(np.int64), "fam": Kf}
            self._surr = self.make_surrogate(pool)
        surr = self._surr
        for a in ("rep", "trust", "public_thr", "ref_norm", "tau", "banned"):
            if hasattr(defense, a) and hasattr(surr, a):
                val = getattr(defense, a); setattr(surr, a, val.clone() if torch.is_tensor(val) else val)
        honest_idx = [i for i in range(N) if i not in set(self.mal)]
        mu = hu.mean(0); sd = torch.clamp(hu.std(0) if m > 1 else hu.abs() * 0.2, min=1e-8)
        g = torch.Generator(device=dev).manual_seed(int(rng.integers(1 << 30)))
        prox = hu[torch.randint(0, m, (len(honest_idx),), generator=g, device=dev)] + noise * sd * torch.randn(len(honest_idx), hu.shape[1], generator=g, device=dev)
        cs = Ctx(ctx.round, ctx.theta, ctx.shapes, N, ctx.beacon)

        def build(mal_rows):
            U = torch.zeros(N, hu.shape[1], device=dev); U[honest_idx] = prox; U[self.mal] = mal_rows
            return U

        def aggregate(mal_rows):
            return copy.deepcopy(surr).aggregate(build(mal_rows), cs)

        def recall_fpr(delta):
            th = ctx.theta + delta
            return float((M.predict(th, ctx.shapes, Xt)[0] == 1).float().mean()), float((M.predict(th, ctx.shapes, Xb)[0] == 1).float().mean())
        return dict(surr=surr, cs=cs, build=build, aggregate=aggregate, recall_fpr=recall_fpr, mu=mu, sd=sd, N=N, m=m)

    def _opt_es(self, honest_U, hu, pu, ctx, defense, tenants, mi, gens: int = 4, pop: int = 6):
        """Independent search: a (mu, lambda) evolution strategy over SIX update directions (poison push, sign flip, mean shift,
        inverse standard deviation, inverse sign, LIE target), same objective as _opt_adaptive and the same number of
        evaluations of the private rule copy (gens x pop = 24)."""
        S = self._sim_setup(hu, ctx, defense, tenants)
        if S is None:
            return hu
        rng = self.rng; mu, sd = S["mu"], S["sd"]; nh = hu.norm(dim=1, keepdim=True)
        n_, m_ = self.n, len(self.mal); s_ = int(np.floor(n_ / 2 + 1) - m_)
        z = float(norm.ppf(min(max((n_ - s_) / n_, 0.5001), 0.9999)))
        dirs = [pu - hu, -hu, (-mu / (mu.norm() + 1e-12)) * nh, (-sd / (sd.norm() + 1e-12)) * nh,
                (-torch.sign(mu) / np.sqrt(mu.numel())) * nh, (mu - z * sd).unsqueeze(0) - hu]
        rows = lambda a: hu + sum(float(a[k]) * dirs[k] for k in range(6))
        rec0, fpr0 = S["recall_fpr"](S["aggregate"](hu))

        def J(a):
            rec, fpr = S["recall_fpr"](S["aggregate"](rows(a)))
            return (rec0 - rec) - 0.5 * max(0.0, fpr - fpr0 - 0.02)
        mean = np.array(self._prev_a if (self._prev_a is not None and len(self._prev_a) == 6) else np.zeros(6), dtype=float)
        sigma, best, bj = 1.0, np.zeros(6), 0.0
        for _ in range(gens):
            cand = [np.clip(mean + sigma * rng.normal(size=6), 0.0, 8.0) for _ in range(pop)]
            scores = [J(a) for a in cand]
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > bj:
                best, bj = cand[order[0]], scores[order[0]]
            mean = np.mean([cand[i] for i in order[:2]] + [best], axis=0); sigma *= 0.6
        self._prev_a = tuple(float(x) for x in best); self.last_gamma = float(best[0])
        out = rows(best); S["surr"].aggregate(S["build"](out), S["cs"])
        return out

    def _agr_tailored(self, honest_U, hu, pu, ctx, defense, tenants, mi):
        """AGR-tailored attack of Shejwalkar and Houmansadr (NDSS 2021) with partial knowledge, run against the attacker's
        private copy of the published rule: every malicious tenant submits mu + gamma * p for a perturbation p in
        {inverse unit vector, inverse standard deviation, inverse sign}; gamma and p are chosen to push the aggregate as far
        as possible from the benign aggregate (untargeted objective, as in the original). 3 directions x 8 gammas = 24."""
        S = self._sim_setup(hu, ctx, defense, tenants)
        if S is None:
            return hu
        mu, sd = S["mu"], S["sd"]; m_ = len(self.mal)
        ref = S["aggregate"](hu)
        P = [(-mu / (mu.norm() + 1e-12), float(mu.norm())), (-sd, 1.0), (-torch.sign(mu) * float(mu.norm()) / np.sqrt(mu.numel()), 1.0)]
        best, bd = hu, -1.0
        for p, g0 in P:
            for k in range(-3, 5):
                u = (mu + g0 * (2.0 ** k) * p).unsqueeze(0).repeat(m_, 1)
                dv = float((S["aggregate"](u) - ref).norm())
                if dv > bd:
                    best, bd = u, dv
        S["surr"].aggregate(S["build"](best), S["cs"])
        return best

    # --------------------------------------------------------------- teach-to-the-test
    def _memorize(self, hu, pu, ctx, defense, tenants):
        """Poison, then patch the probe.

        The attacker trains on its own data with target-family flows relabelled benign (the poison) and ALSO on the
        probe samples it knows, with their TRUE labels, for longer than an honest tenant. Against a public rule the
        known probe is the exact one, so the update can keep the probe check clean while still suppressing DoS flows
        that are not in the probe. Against a sealed rule it knows only a stand-in (its own samples plus any leaked
        pool samples), so the patch does not transfer to the probe drawn at reveal time."""
        from . import model as M

        v = defense.public_view()
        if defense.public:
            idx = v["probe_idx"]
            Kx, Kf = v["pool_X"][idx], v["pool_fam"][idx]
        else:
            Kx, Kf = self._surrogate(tenants, defense, ctx)
        dev = ctx.theta.device
        Kx = Kx.to(dev) if torch.is_tensor(Kx) else torch.as_tensor(Kx, device=dev)
        Ky = torch.as_tensor((np.asarray(Kf) != "benign").astype(np.int64), device=dev)
        Xs, Ys = [], []
        rep = max(1, int(round(0.5 * len(tenants[self.mal[0]]["y"]) / max(len(Ky), 1))))
        for i in self.mal:
            t = tenants[i]
            x = torch.as_tensor(t["X"], device=dev)
            y = torch.as_tensor(t["y"], device=dev).clone()
            y[torch.as_tensor(t["fam"] == self.target, device=dev)] = 0          # the poison
            Xs.append(torch.cat([x] + [Kx] * rep)); Ys.append(torch.cat([y] + [Ky] * rep))
        X, Y = torch.stack(Xs), torch.stack(Ys)
        tr = self.train
        start = ctx.theta + pu.mean(0)                                           # begin from the poisoned model
        with torch.enable_grad():                                                # craft() runs under no_grad
            th = M.local_train(start, ctx.shapes, X, Y, max(8, 4 * tr["epochs"]), tr["bs"], tr["lr"], tr["gen"])
        return th - ctx.theta.unsqueeze(0)

    # --------------------------------------------------------------- rule-aware search
    def _rule_aware(self, honest_U, hu, pu, ctx, defense, tenants, benign_U, mi):
        d = pu - hu                                           # push toward the goal
        ref = self._proxies(honest_U, mi, benign_U).mean(0)   # attacker's estimate of the benign mean
        name = defense.name

        if name in D.PROBE_NAMES:
            v = defense.public_view()
            if defense.public:                                # exact probe is readable on chain
                idx = v["probe_idx"]
                Xp, fp = v["pool_X"][idx], v["pool_fam"][idx]
            else:                                             # sealed: only a surrogate from its own data (+ any leaked samples)
                Xp, fp = self._surrogate(tenants, defense, ctx)
            mode = defense.mode
            if mode == "abs":
                thr = defense.public_thr                      # published absolute tolerance

                def passes(gam):
                    dmg, _ = D.damage(ctx.theta, hu + gam * d, ctx.shapes, Xp, fp, defense.families)
                    return bool((dmg <= self.margin * thr).all())
            elif mode == "rel_max":
                # strongest realistic attacker: it knows the rule FORM and its constants (kappa, floor) because they
                # are on chain, and estimates the benign median/MAD of harm from its own honest updates.
                d0, _ = D.damage(ctx.theta, hu, ctx.shapes, Xp, fp, defense.families)
                med = d0.median(); mad = (d0 - med).abs().median()
                thr = float(max(med + defense.kappa * (mad + 1e-3) + defense.delta_floor, d0.max()))

                def passes(gam):
                    dmg, _ = D.damage(ctx.theta, hu + gam * d, ctx.shapes, Xp, fp, defense.families)
                    return bool((dmg <= thr).all())
            else:                                             # rel_fam: peer test per check
                Rh, fh = D.probe_stats(ctx.theta, hu, ctx.shapes, Xp, fp, defense.families)
                _, ref = D.peer_z(Rh[1:], fh[1:], defense.s_min)   # its own honest updates stand in for the peers

                def passes(gam):
                    R, f_ = D.probe_stats(ctx.theta, hu + gam * d, ctx.shapes, Xp, fp, defense.families)
                    z, _ = D.peer_z(R[1:], f_[1:], defense.s_min, ref=ref)
                    return bool((z.max(1).values <= self.margin * defense.kappa).all())
        elif name == "normgate":
            tau = defense.public_view()["tau"]
            if not np.isfinite(tau):
                tau = 1.5 * ref.norm().item()

            def passes(gam):
                return bool(((hu + gam * d).norm(dim=1) <= self.margin * tau).all())
        elif name == "cosrep":
            def passes(gam):
                u = hu + gam * d
                return bool((D._cos(u, ref) > 0.35).all())    # stays well above the reputation floor
        elif name == "trustaware":
            prox = self._proxies(honest_U, mi, benign_U)
            dmax = (prox - ref).norm(dim=1).max()

            def passes(gam):
                return bool(((hu + gam * d) - ref).norm(dim=1).max() <= self.margin * dmax * 1.5)
        else:                                                  # fedavg / median / trmean / mkrum
            prox = self._proxies(honest_U, mi, benign_U)
            dmax = torch.cdist(prox, prox).max()

            def passes(gam):
                u = hu + gam * d
                return bool((u.unsqueeze(1) - prox.unsqueeze(0)).norm(dim=2).max() <= dmax * 1.0)

        lo, hi = 0.0, self.gamma_max
        if passes(hi):
            lo = hi
        else:
            for _ in range(14):
                g = (lo + hi) / 2
                if passes(g):
                    lo = g
                else:
                    hi = g
        self.last_gamma = lo
        return hu + lo * d
