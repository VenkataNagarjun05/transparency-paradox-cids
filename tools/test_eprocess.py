"""Computational check of the e-process slashing rules (qbfl/accountability.py) on synthetic honest tenants.

H0  exchangeable : every tenant's badness score is i.i.d. N(0,1) in every round. Every anytime-valid rule must slash at most
                   alpha of the tenants (Ville). This checks the implementation.
H-het            : persistent tenant differences (non-IID data): score = mu_i + noise, mu_i ~ N(0, 1). Violates exchangeability
                   ACROSS tenants (peer p-values), not over time.
H-drift          : common non-stationarity: score = noise * (1 + 2 t/T) + trend, shared by all tenants. Violates exchangeability
                   OVER TIME (self / raw p-values), not across tenants.
H-hetdrift       : tenant-specific drift: score = noise + b_i * t/T, b_i ~ N(0, 1). Violates both.
Usage: python3 tools/test_eprocess.py  -> prints the honest-slash rate per rule and alpha for each scenario
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qbfl.accountability import Accountability  # noqa: E402


class Stub:
    """Minimal stand-in for a defense: exposes this round's scores as last['dmg'] and no threshold."""
    def __init__(self):
        self.last = {}


def run(scenario: str, n=20, T=40, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    hits = {}
    for rep in range(reps):
        acct = Accountability(n, seed=rep)
        d = Stub()
        mu = rng.normal(0, 1, n); b = rng.normal(0, 1, n)
        for t in range(T):
            e = rng.normal(0, 1, n)
            if scenario == "H0":
                s = e
            elif scenario == "het":
                s = mu + e
            elif scenario == "drift":
                s = e * (1 + 2 * t / T) + 3 * t / T
            else:
                s = e + 3 * b * t / T
            d.last = {"dmg": s, "thr": float("inf")}
            acct.update(t, np.zeros((n, 1)), d)
        for k, at in acct.slashed_at.items():
            if "@" in k:
                hits.setdefault(k, []).append(np.mean(at >= 0))
    return {k: float(np.mean(v)) for k, v in hits.items()}


def main():
    import torch  # noqa: F401  (accountability imports torch types)
    rows = {}
    for sc in ("H0", "het", "drift", "hetdrift"):
        rows[sc] = run(sc)
    rules = sorted(rows["H0"])
    print(f"{'rule':16s}" + "".join(f"{sc:>10s}" for sc in rows))
    for r in rules:
        a = float(r.split("@")[1])
        flag = "" if rows["H0"][r] <= a * 1.1 + 0.005 else "   <-- H0 VIOLATION"
        print(f"{r:16s}" + "".join(f"{rows[sc][r]:10.3f}" for sc in rows) + flag)
    import json
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "results", "eprocess_synthetic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
