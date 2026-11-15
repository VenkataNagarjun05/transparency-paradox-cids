"""Unit tests for the qbfl package (run: python3 -m pytest -q tests/test_core.py)."""
import os
import struct
import sys
import hashlib

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qbfl import data as D, pqc  # noqa: E402
from qbfl.accountability import Accountability  # noqa: E402


def _world(scaler_fit="train", holdout=None):
    _, fp = D.split_pools(D.synthetic_pools(n_per=4000, seed=1))
    return fp, D.build_world(fp, n_tenants=8, n_train=400, n_test=600, n_audit=600, seed=0, scaler_fit=scaler_fit, holdout_source=holdout)


def test_scaler_is_fitted_on_tenant_training_rows_only():
    fp, w = _world("train")
    T = np.concatenate([t["X"] for t in w.tenants])
    # tenant rows are standardized by construction (up to resampling duplicates and clipping)...
    assert abs(T.mean()) < 0.05 and abs(T.std() - 1) < 0.1
    # ...and the selection of rows does not depend on the scaler mode (only the fit does)
    _, w_all = _world("all")
    assert all(np.array_equal(a["y"], b["y"]) for a, b in zip(w.tenants, w_all.tenants))
    assert not np.allclose(w.test["src0"]["X"], w_all.test["src0"]["X"])


def test_holdout_source_never_enters_tenants_or_probe_pool():
    _, w = _world(holdout="src2")
    assert all(t["source"] != "src2" for t in w.tenants)
    assert "src2" not in w.audit and "src2" in w.test


def test_commitment_layout_matches_the_contract():
    delta = np.arange(5, dtype=np.float32); salt = b"\x07" * 16
    expected = hashlib.sha256(struct.pack(">II", 3, 9) + salt + hashlib.sha256(delta.tobytes()).digest()).digest()
    assert pqc.commitment(3, 9, delta, salt) == expected


def test_calldata_gas_follows_eip2028():
    assert pqc.calldata_gas(b"\x00\x00\x01\xff") == 4 * 2 + 16 * 2


@pytest.mark.skipif(not pqc.HAVE_OQS, reason="liboqs not installed")
def test_post_quantum_attack_checks_all_rejected():
    res = pqc.security_checks()
    flat = {k: v for k, v in res.items() if isinstance(v, bool)}
    assert flat and all(flat.values()), res


class _Stub:
    last = {}


@pytest.mark.parametrize("rule", ["peer_ctm", "self_ctm", "raw_ctm", "peer_bet", "self_bet", "raw_bet"])
def test_eprocess_rules_keep_alpha_under_exchangeable_honest_scores(rule):
    rng = np.random.default_rng(0); n, T, reps = 20, 40, 80
    hits = {a: [] for a in Accountability.ALPHAS}
    for rep in range(reps):
        acct = Accountability(n, seed=rep); d = _Stub()
        for t in range(T):
            d.last = {"dmg": rng.normal(size=n), "thr": float("inf")}
            acct.update(t, np.zeros((n, 1)), d)
        for a in Accountability.ALPHAS:
            hits[a].append(np.mean(acct.slashed_at[f"{rule}@{a}"] >= 0))
    for a, h in hits.items():
        assert np.mean(h) <= a + 0.02, (rule, a, np.mean(h))
