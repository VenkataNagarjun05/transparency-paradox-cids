"""One federated run: tenants, a defense (contract rule), an optional attacker."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import model as M
from .attacks import Attacker
from .defenses import Ctx, make_defense, PROBE_NAMES
from .data import World, FAMILIES
from .accountability import Accountability


@dataclass
class Config:
    defense: str = "fedavg"
    attack: str = "none"
    mal_frac: float = 0.3
    seed: int = 0
    rounds: int = 40
    local_epochs: int = 2
    bs: int = 128
    lr: float = 2e-3
    target: str = "dos"
    attack_start: int = 5
    farm_rounds: int = 10
    probe_size: int = 1024
    leak_frac: float = 0.0
    see_benign: bool = False         # True = no commit-reveal: attacker sees this round's benign updates
    reuse_probes: bool = False       # sealed rule: reuse revealed probes instead of burning them
    use_rep: bool = True
    hidden: tuple = (128, 64)
    eval_every: int = 5
    dparams: dict = field(default_factory=dict)   # defense hyper-parameters (tuned values go here)
    extra: dict = field(default_factory=dict)

    def run_id(self):
        d = asdict(self)
        d.pop("extra")
        d["dparams"] = json.dumps(self.dparams, sort_keys=True)
        return hashlib.md5(repr(sorted(d.items())).encode()).hexdigest()[:12]


def _stack(world: World, device):
    X = torch.as_tensor(np.stack([t["X"] for t in world.tenants]), device=device)
    y = torch.as_tensor(np.stack([t["y"] for t in world.tenants]), device=device)
    fam = np.stack([t["fam"] for t in world.tenants])
    return X, y, fam


def _eval(theta, shapes, world: World, target, device):
    """Overall and per-source metrics, including AUC from attack-class probabilities."""
    out = {}
    Ps, Ss, ys, fs = [], [], [], []
    for s, d in world.test.items():
        X = torch.as_tensor(d["X"], device=device)
        pred = M.predict(theta, shapes, X)[0].cpu().numpy()
        sc = M.predict_proba(theta, shapes, X)[0].cpu().numpy()
        ms = M.metrics(pred, d["y"], d["fam"], target, score=sc)
        out[f"f1_{s}"] = ms["macro_f1"]
        out[f"rec_target_{s}"] = ms.get("rec_target", float("nan"))
        out[f"fpr_{s}"] = ms["fpr"]
        out[f"auc_{s}"] = ms.get("auc", float("nan"))
        for k_, v_ in ms.items():                               # per-family recall on this source (generalisation diagnostics)
            if k_.startswith("rec_") and k_ != "rec_target":
                out[f"{k_}_{s}"] = v_
        Ps.append(pred); Ss.append(sc); ys.append(d["y"]); fs.append(d["fam"])
    out.update(M.metrics(np.concatenate(Ps), np.concatenate(ys), np.concatenate(fs), target, score=np.concatenate(Ss)))
    return out


@torch.no_grad()
def mia_auc(theta, shapes, world: World, device, n: int = 3000, seed: int = 0) -> float:
    """Loss-threshold membership inference: AUC of telling training rows (members) from test rows by the
    global model's loss. 0.5 = no leakage; higher = the model memorises its training tenants' data."""
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    Xm = np.concatenate([t["X"] for t in world.tenants]); ym = np.concatenate([t["y"] for t in world.tenants])
    Xn = np.concatenate([d["X"] for d in world.test.values()]); yn = np.concatenate([d["y"] for d in world.test.values()])
    im = rng.choice(len(Xm), min(n, len(Xm)), replace=False); inn = rng.choice(len(Xn), min(n, len(Xn)), replace=False)
    def loss(X, y):
        X = torch.as_tensor(X, device=device); y = torch.as_tensor(y, device=device)
        lg = M.forward(theta.unsqueeze(0), shapes, X)[0]
        return F.cross_entropy(lg, y, reduction="none").cpu().numpy()
    lm, ln = loss(Xm[im], ym[im]), loss(Xn[inn], yn[inn])
    return float(roc_auc_score(np.r_[np.ones(len(lm)), np.zeros(len(ln))], -np.r_[lm, ln]))


def run_reference(world: World, cfg, device="cpu", mode="centralized", epochs: int = 20) -> dict:
    """Reference points for the federated results.
    centralized: one model trained on all tenants' data pooled (upper bound, no privacy).
    local_only : every tenant trains alone on its own data; metrics averaged over tenants (lower bound)."""
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    shapes = M.layer_shapes(world.n_features, cfg.hidden, 2)
    theta = M.init_theta(shapes, cfg.seed, device)
    X, y, _ = _stack(world, device)
    if mode == "centralized":
        Xc = X.reshape(1, -1, X.shape[-1]); yc = y.reshape(1, -1)
        th = M.local_train(theta, shapes, Xc, yc, epochs, cfg.bs * 4, cfg.lr, gen)[0]
        final = _eval(th, shapes, world, cfg.target, device)
    else:
        th = M.local_train(theta, shapes, X, y, epochs, cfg.bs, cfg.lr, gen)
        evs = [_eval(th[i], shapes, world, cfg.target, device) for i in range(th.shape[0])]
        final = {k: float(np.nanmean([e[k] for e in evs])) for k in evs[0]}
    return dict(run_id=f"{mode}-{cfg.seed}-{cfg.target}-{cfg.hidden}", defense=mode, attack="none", mal_frac=0.0,
                seed=cfg.seed, target=cfg.target, hidden=str(cfg.hidden), dparams="{}", final=final, curve=[final])


def pick_malicious(world: World, frac: float, target: str, rng) -> list:
    n = len(world.tenants)
    m = int(round(frac * n))
    if m == 0:
        return []
    share = np.array([np.mean(t["fam"] == target) for t in world.tenants])
    elig = np.flatnonzero(share >= 0.05)             # attacker must actually hold target-family traffic
    if len(elig) < m:
        elig = np.argsort(-share)[: max(m, 1)]
    return sorted(rng.choice(elig, m, replace=False).tolist())


def run_one(world: World, cfg: Config, device="cpu", verbose=False, hooks=None) -> dict:
    """hooks (optional): object with before(r, U_sub, mal) -> beacon bytes or None, and after(r, theta, defense, mal).
    Used by chain/integration.py to drive the on-chain round; experiments run with hooks=None."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    X, y, fam = _stack(world, device)
    C = X.shape[0]
    shapes = M.layer_shapes(world.n_features, cfg.hidden, 2)
    theta = M.init_theta(shapes, cfg.seed, device)

    pool = {k: np.concatenate([world.audit[s][k] for s in world.audit]) for k in ("X", "y", "fam")}
    dkw = dict(cfg.dparams)
    if cfg.defense in PROBE_NAMES:
        dkw = {**{"probe_size": cfg.probe_size, "leak_frac": cfg.leak_frac, "reuse": cfg.reuse_probes,
                  "use_rep": cfg.use_rep}, **dkw}
    defense = make_defense(cfg.defense, C, device, audit_pool=pool, families=FAMILIES, seed=cfg.seed, **dkw)
    mal = pick_malicious(world, cfg.mal_frac, cfg.target, rng) if cfg.attack != "none" else []
    attacker = Attacker(cfg.attack, mal, C, farm_rounds=cfg.farm_rounds, see_benign=cfg.see_benign, seed=cfg.seed) if mal else None
    if attacker is not None:
        attacker.target = cfg.target
        attacker.train = dict(epochs=cfg.local_epochs, bs=cfg.bs, lr=cfg.lr, gen=gen)
        attacker.families = FAMILIES
        attacker.make_surrogate = lambda pool: make_defense(cfg.defense, C, device, audit_pool=pool, families=FAMILIES, seed=cfg.seed + 777, **dkw)
    y_poison = y.clone()
    tmask = torch.as_tensor(fam == cfg.target, device=device)
    y_poison[tmask] = 0                                    # target-family flows relabelled benign
    mal_t = torch.as_tensor(mal, device=device, dtype=torch.long) if mal else None
    acct = Accountability(C, alpha=cfg.extra.get('acct_alpha', 0.05), seed=cfg.seed)
    if cfg.extra.get("acct_trace"):
        acct.trace = []
    beacon = b"genesis"
    curve, mass_mal, rej_hon, rej_mal = [], [], [], []
    honest_idx = [i for i in range(C) if i not in set(mal)]
    t_def = 0.0

    for r in range(cfg.rounds):
        th_loc = M.local_train(theta, shapes, X, y, cfg.local_epochs, cfg.bs, cfg.lr, gen)
        U = th_loc - theta.unsqueeze(0)
        attacking = attacker is not None and r >= cfg.attack_start
        if attacking:
            th_p = M.local_train(theta, shapes, X[mal_t], y_poison[mal_t], cfg.local_epochs, cfg.bs, cfg.lr, gen)
            P = th_p - theta.unsqueeze(0)
            beacon = hashlib.sha256(beacon + str(r).encode()).digest()
            ctx = Ctx(r, theta, shapes, C, beacon)
            benign_U = U[torch.as_tensor(honest_idx, device=device)] if cfg.see_benign else None
            U_sub = U.clone()
            U_sub[mal_t] = attacker.craft(U, P, ctx, defense, world.tenants, benign_U)
        else:
            beacon = hashlib.sha256(beacon + str(r).encode()).digest()
            ctx = Ctx(r, theta, shapes, C, beacon)
            U_sub = U
        if hooks is not None:
            b = hooks.before(r, U_sub, mal)
            if b is not None:
                ctx = Ctx(r, theta, shapes, C, b)       # probes drawn from the ON-CHAIN beacon
        t1 = time.time()
        delta = defense.aggregate(U_sub, ctx)
        t_def += time.time() - t1
        theta = theta + delta
        acct.update(r, U_sub, defense)
        if hooks is not None:
            hooks.after(r, theta, defense, mal)
        w = defense.last_w
        if w is not None and w.sum() > 0 and mal:
            mass_mal.append(float(w[mal_t].sum() / w.sum()))
            rej_mal.append(float((w[mal_t] <= 0).float().mean()))
            rej_hon.append(float((w[torch.as_tensor(honest_idx, device=device)] <= 0).float().mean()))
        elif w is not None and w.sum() > 0:
            rej_hon.append(float((w <= 0).float().mean()))
        if (r + 1) % cfg.eval_every == 0 or r == cfg.rounds - 1:
            e = _eval(theta, shapes, world, cfg.target, device)
            e["round"] = r + 1
            curve.append(e)
            if verbose:
                print(r + 1, {k: round(v, 3) for k, v in e.items() if k in ("macro_f1", "rec_target", "fpr")})
    final = curve[-1]
    row = dict(run_id=cfg.run_id(), **{k: v for k, v in asdict(cfg).items() if k not in ("extra", "hidden", "dparams")},
               dparams=json.dumps(cfg.dparams, sort_keys=True),
               n_mal=len(mal), final=final, curve=curve,
               mass_on_mal=float(np.mean(mass_mal)) if mass_mal else float("nan"),
               mal_rejected=float(np.mean(rej_mal)) if rej_mal else float("nan"),
               honest_rejected=float(np.mean(rej_hon)) if rej_hon else float("nan"),
               defense_sec=t_def, wall_sec=time.time() - t0,
               banned=int(defense.banned.sum()), last_gamma=getattr(attacker, "last_gamma", float("nan")),
               hidden=str(cfg.hidden), sec_per_round=(time.time() - t0) / max(cfg.rounds, 1),
               mia_auc=mia_auc(theta, shapes, world, device, seed=cfg.seed),
               **acct.summary(mal, cfg.attack_start))
    if acct.trace is not None:
        row["acct_trace"] = dict(mal=list(map(int, mal)), rounds=acct.trace)
    return row
