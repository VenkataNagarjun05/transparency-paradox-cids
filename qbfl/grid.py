"""Experiment grids with resume. Each finished run is one JSON line, so a Kaggle session that dies
can be restarted and continues where it stopped."""
from __future__ import annotations

import itertools
import json
import os
import time

from .fl import Config, run_one

DEFENSES = ["fedavg", "median", "trmean", "mkrum", "normgate", "cosrep", "trustaware", "probe_public", "sealed"]
FINAL_DEFENSES = ["fedavg", "median", "trmean", "mkrum", "normgate", "cosrep", "trustaware",
                  "probe_public", "sealed_abs", "sealed", "public_fam", "sealed_fam"]
TUNE_SEEDS = (100, 101, 102)          # never used for the final numbers
FINAL_SEEDS = tuple(range(10))

_Q = (0.9, 0.95, 0.99)
TUNE_GRIDS = {
    "fedavg": [{}], "median": [{}],
    "trmean": [{"beta": b} for b in (0.1, 0.2, 0.3, 0.4)],
    "mkrum": [{"f_frac": f} for f in (0.1, 0.2, 0.3)],
    "normgate": [{"tau_mult": t} for t in (1.0, 1.25, 1.5, 2.0, 3.0)],
    "cosrep": [{"lam": l, "rho": r} for l in (0.5, 0.8) for r in (0.1, 0.2, 0.3)],
    "trustaware": [{"lam": l, "temp": t} for l in (0.5, 0.7) for t in (1.0, 2.0, 4.0)],
    "probe_public": [{"calib_q": q, "delta_floor": f} for q in _Q for f in (0.0, 0.02, 0.05)],
    "sealed_abs": [{"calib_q": q, "delta_floor": f} for q in _Q for f in (0.0, 0.02, 0.05)],
    "sealed": [{"kappa": k, "delta_floor": f, "use_rep": r} for k in (2.0, 3.0, 5.0) for f in (0.02, 0.05, 0.1) for r in (True, False)],
    "public_fam": [{"kappa": k, "s_min": m, "use_rep": r} for k in (2.0, 3.0, 4.0, 6.0) for m in (0.02, 0.05, 0.1) for r in (True, False)],
    "sealed_fam": [{"kappa": k, "s_min": m, "use_rep": r} for k in (2.0, 3.0, 4.0, 6.0) for m in (0.02, 0.05, 0.1) for r in (True, False)],
}
TUNE_ATTACKS = ["none", "rule_aware", "rule_aware_farm"]
ATTACKS = ["none", "targeted_lf", "signflip", "lie", "minmax", "rule_aware", "rule_aware_farm"]


def preset(name: str, seeds=(0, 1, 2)):
    cfgs = []
    if name == "gate":          # day-7 go / no-go: does the rule-aware attack beat the public rules, does sealing stop it
        for d, a, s in itertools.product(DEFENSES, ["none", "targeted_lf", "rule_aware", "rule_aware_farm"], seeds):
            cfgs.append(Config(defense=d, attack=a, mal_frac=0.3, seed=s))
    elif name == "main":
        for d, a, f, s in itertools.product(DEFENSES, ATTACKS, [0.1, 0.2, 0.3, 0.4], seeds):
            if a == "none" and f != 0.3:
                continue
            cfgs.append(Config(defense=d, attack=a, mal_frac=f, seed=s))
    elif name == "ablation":    # what each piece of the sealed rule buys
        for s in seeds:
            for kw in [dict(), dict(reuse_probes=True), dict(use_rep=False), dict(see_benign=True)]:
                for a in ["rule_aware", "rule_aware_farm"]:
                    cfgs.append(Config(defense="sealed", attack=a, mal_frac=0.3, seed=s, **kw))
            for lk in [0.01, 0.05, 0.2, 0.5]:   # how much probe-pool leakage breaks sealing
                cfgs.append(Config(defense="sealed", attack="rule_aware", mal_frac=0.3, seed=s, leak_frac=lk))
    else:
        raise KeyError(name)
    return cfgs


def tune_preset(seeds=TUNE_SEEDS, mal_frac=0.3):
    """Same budget for every defense: each candidate setting is tried on the same attacks and seeds."""
    cfgs = []
    for d, grid_ in TUNE_GRIDS.items():
        for dp in grid_:
            for a in TUNE_ATTACKS:
                for s in seeds:
                    cfgs.append(Config(defense=d, attack=a, mal_frac=mal_frac, seed=s, dparams=dp))
    return cfgs


def final_preset(best: dict, seeds=FINAL_SEEDS, fracs=(0.1, 0.2, 0.3, 0.4),
                 attacks=("targeted_lf", "lie", "minmax", "rule_aware", "rule_aware_farm")):
    """Final study: tuned settings, held-out seeds."""
    cfgs = []
    for d in FINAL_DEFENSES:
        dp = best.get(d, {})
        for s in seeds:
            cfgs.append(Config(defense=d, attack="none", mal_frac=0.3, seed=s, dparams=dp))
            for a in attacks:
                for f in fracs:
                    cfgs.append(Config(defense=d, attack=a, mal_frac=f, seed=s, dparams=dp))
    return cfgs


MEM_SEEDS = tuple(range(10, 20))        # fresh seeds, never used for tuning or for the first final study
MEM_DEFENSES = ["fedavg", "median", "trmean", "cosrep", "probe_public", "sealed_abs", "public_fam", "sealed_fam"]


def memorize_preset(best: dict, seeds=MEM_SEEDS, fracs=(0.2, 0.3, 0.4)):
    """POST-HOC follow-up (added after seeing the first final results): an attacker that memorizes the probe set it knows.
    Defense settings are the ones tuned on data half A; nothing is re-tuned."""
    cfgs = []
    for d in MEM_DEFENSES:
        dp = best.get(d, {})
        for s in seeds:
            cfgs.append(Config(defense=d, attack="none", mal_frac=0.3, seed=s, dparams=dp))
            for a in ("rule_aware", "probe_memorize"):
                for f in fracs:
                    cfgs.append(Config(defense=d, attack=a, mal_frac=f, seed=s, dparams=dp))
    for d in ("sealed_fam", "sealed_abs"):                    # how much probe-pool leakage breaks sealing
        for s in seeds:
            for lk in (0.01, 0.05, 0.2, 0.5):
                cfgs.append(Config(defense=d, attack="probe_memorize", mal_frac=0.3, seed=s, dparams=best.get(d, {}), leak_frac=lk))
    return cfgs


SETTLE_SEEDS = tuple(range(20, 50))      # 30 fresh seeds, confirmatory
STRONG_SEEDS = tuple(range(50, 60))      # 10 fresh seeds for the optimising attacker
SETTLE_DEFENSES = ["probe_public", "public_fam", "sealed_abs", "sealed_fam"]
STRONG_DEFENSES = ["fedavg", "median", "trmean", "mkrum", "normgate", "cosrep", "trustaware",
                   "probe_public", "sealed_abs", "public_fam", "sealed_fam"]


def settle_preset(best: dict, seeds=SETTLE_SEEDS):
    """Confirmatory: sealed vs public probes under the memorizing attacker. Primary cell = probe_memorize at 30%."""
    cfgs = []
    for d in SETTLE_DEFENSES:
        dp = best.get(d, {})
        for s in seeds:
            cfgs.append(Config(defense=d, attack="none", mal_frac=0.3, seed=s, dparams=dp))
            cfgs.append(Config(defense=d, attack="rule_aware", mal_frac=0.3, seed=s, dparams=dp))
            for f in (0.2, 0.3, 0.4):
                cfgs.append(Config(defense=d, attack="probe_memorize", mal_frac=f, seed=s, dparams=dp))
    return cfgs


def strong_preset(best: dict, seeds=STRONG_SEEDS):
    """Defense-agnostic optimising attacker against every rule, with the older attackers on the same seeds for pairing."""
    cfgs = []
    for d in STRONG_DEFENSES:
        dp = best.get(d, {})
        for s in seeds:
            cfgs.append(Config(defense=d, attack="none", mal_frac=0.3, seed=s, dparams=dp))
            for a in ("rule_aware", "probe_memorize"):
                cfgs.append(Config(defense=d, attack=a, mal_frac=0.3, seed=s, dparams=dp))
            for f in (0.3, 0.4):
                cfgs.append(Config(defense=d, attack="opt_adaptive", mal_frac=f, seed=s, dparams=dp))
    return cfgs


# ------------------------------------------------------------------ Phase 2 (robustness checks, fresh seeds, tuned settings)
P2_DEFENSES = ["fedavg", "median", "trmean", "cosrep", "probe_public", "sealed_fam"]
ACCT_DEFENSES = ["probe_public", "sealed_abs", "public_fam", "sealed_fam", "cosrep", "median"]
ACCT_ATTACKS = ["none", "rule_aware", "rule_aware_farm", "probe_memorize", "opt_adaptive"]


def p2_preset(best: dict, defenses, attacks, seeds, fracs=(0.3,), **kw):
    cfgs = []
    for d in defenses:
        dp = best.get(d, {})
        for s in seeds:
            for a in attacks:
                for f in (fracs if a != "none" else (0.3,)):
                    cfgs.append(Config(defense=d, attack=a, mal_frac=f, seed=s, dparams=dp, **kw))
    return cfgs


def ablation_preset(best: dict, seeds=FINAL_SEEDS):
    """What each mechanism buys, and how much leakage or a missing commit-reveal step costs."""
    cfgs = []
    for s in seeds:
        for a in ("rule_aware", "rule_aware_farm"):
            for d in ("sealed_fam", "sealed"):
                dp = best.get(d, {})
                for kw in (dict(reuse_probes=True), dict(see_benign=True), dict(use_rep=False)):
                    cfgs.append(Config(defense=d, attack=a, mal_frac=0.3, seed=s, dparams={k: v for k, v in dp.items() if k != "use_rep"} if "use_rep" in kw else dp, **kw))
            for lk in (0.01, 0.05, 0.2, 0.5):                 # probe-pool leakage to the attacker
                cfgs.append(Config(defense="sealed_fam", attack=a, mal_frac=0.3, seed=s, dparams=best.get("sealed_fam", {}), leak_frac=lk))
    return cfgs


def run_grid(world, cfgs, out_path, device="cuda", tag=""):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["run_id"])
            except Exception:
                pass
    todo = [c for c in cfgs if c.run_id() not in done]
    print(f"{len(cfgs)} configs, {len(done)} done, {len(todo)} to run")
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, c in enumerate(todo):
            try:
                row = run_one(world, c, device)
            except Exception as e:  # keep the sweep alive, record the failure
                row = dict(run_id=c.run_id(), error=repr(e), defense=c.defense, attack=c.attack, seed=c.seed)
            row["tag"] = tag
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            if i % 10 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(todo)}] {c.defense:12s} {c.attack:16s} mal={c.mal_frac} seed={c.seed}  {el/60:.1f} min elapsed", flush=True)


def _worker(world, cfgs, out_path, device, tag):
    run_grid(world, cfgs, out_path, device=device, tag=tag)


def run_parallel(world, cfgs, out_prefix, devices, tag=""):
    """One process per GPU. Each process writes its own JSONL shard, so there are no write clashes."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for i, dev in enumerate(devices):
        shard = cfgs[i::len(devices)]
        p = ctx.Process(target=_worker, args=(world, shard, f"{out_prefix}.shard{i}.jsonl", dev, tag))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


# ------------------------------------------------------------------ R3 (pre-registered in docs/PREREG_R3.md): how far the attack result reaches
R3_SEEDS = dict(H1B=tuple(range(410, 420)), MAIN=tuple(range(350, 360)), ARCH=tuple(range(420, 425)), BUDGET8=tuple(range(350, 355)))
R3_BUDGET8_DEFENSES = ["median", "trustaware", "cosrep", "sealed_fam", "public_fam"]


def r3_preset(best: dict, stage: str):
    S, D_ = R3_SEEDS, STRONG_DEFENSES
    if stage == "H1B":        # 10 more seeds for the primary test (clean + optimiser, 30%)
        return p2_preset(best, D_, ["none", "opt_adaptive"], S["H1B"], fracs=(0.3,))
    if stage == "ES":         # independent search procedure, same budget
        return p2_preset(best, D_, ["opt_es"], S["MAIN"], fracs=(0.3,))
    if stage == "AGR":        # published AGR-tailored attack against the simulated rule
        return p2_preset(best, D_, ["agr_tailored"], S["MAIN"], fracs=(0.3,))
    if stage == "FRAC":       # fewer malicious tenants
        return p2_preset(best, D_, ["opt_adaptive"], S["MAIN"], fracs=(0.1, 0.2))
    if stage == "KNOW":       # imperfect knowledge: noisier honest-update estimates; one tenth of own data
        return p2_preset(best, D_, ["opt_noisy", "opt_lowdata"], S["MAIN"], fracs=(0.3,))
    if stage == "BUDGET2":
        return p2_preset(best, D_, ["opt_adaptive_x2"], S["MAIN"], fracs=(0.3,))
    if stage == "ARCH":       # different model and local training: 256-128-64 MLP, 3 local epochs, lr 1e-3
        return p2_preset(best, P2_DEFENSES, ["none", "opt_adaptive"], S["ARCH"], fracs=(0.3,), hidden=(256, 128, 64), local_epochs=3, lr=1e-3)
    if stage == "BUDGET8":
        return p2_preset(best, R3_BUDGET8_DEFENSES, ["opt_adaptive_x8"], S["BUDGET8"], fracs=(0.3,))
    raise KeyError(stage)


# ------------------------------------------------------------------ R4 (docs/PREREG_R4.md): corrected identifier filter, primary comparison re-run
def r4_preset(best: dict):
    return p2_preset(best, STRONG_DEFENSES, ["none", "rule_aware", "probe_memorize", "opt_adaptive"], tuple(range(350, 360)), fracs=(0.3,))
