"""Accountability layer: WHEN to slash a tenant, separate from how updates are aggregated.

Slashing is irreversible and costs the tenant its bond, so it needs a different standard of evidence from
down-weighting one update. Four rules are compared side by side in the same run (observational: they record
who WOULD be slashed and when; they do not change aggregation):

  margin   slash when one round's score exceeds the defense's pass mark by a fixed margin (the naive rule)
  kofm     slash after k offences within the last m rounds (common heuristic)
  peer_ctm anytime-valid e-process on conformal p-values of the tenant's score AMONG ITS PEERS in each round
  self_ctm anytime-valid e-process on conformal p-values of the tenant's peer-rank AGAINST ITS OWN PAST ranks
           ("slash on change, not on difference")
  peer_bet / self_bet  same p-values, bounded betting e-process  E_t = prod (1 + lam (1 - 2 p_s)),  lam = 0.5.
           Also a test supermartingale for super-uniform p (same Ville bound), but each factor is in [0.5, 1.5], so a slightly
           mis-calibrated p-value cannot blow the evidence up; it needs about 7-8 strongly suspicious rounds to reach 1/alpha = 20.

For the two *_ctm rules the running evidence is E_t = prod_s f(p_s) with the calibrator f(p) = k p^(k-1), 0<k<1.
If, for an honest tenant, each p_s is (conditionally) super-uniform, E_t is a test supermartingale and Ville's
inequality gives  P(sup_t E_t >= 1/alpha) <= alpha : an honest tenant is EVER slashed with probability at most alpha.
peer_ctm needs honest tenants to be exchangeable with each other (fails under non-IID data);
self_ctm needs an honest tenant's own peer-rank sequence to be exchangeable over time (stationary behaviour).
Whether these hold on real data is measured, not assumed.
"""
from __future__ import annotations

import numpy as np
import torch


def badness_scores(U: torch.Tensor, defense) -> np.ndarray:
    """Per-tenant badness this round (larger = more suspicious). Uses the defense's own score when it has one,
    otherwise a defense-agnostic one (distance to the coordinate-wise median, scaled by its median)."""
    last = getattr(defense, "last", None) or {}
    if "dmg" in last and len(last["dmg"]) == U.shape[0]:
        return np.asarray(last["dmg"], dtype=float)
    med = U.median(0).values
    d = (U - med).norm(dim=1)
    return (d / (d.median() + 1e-12)).detach().cpu().numpy().astype(float)


class Accountability:
    ALPHAS = (0.01, 0.05, 0.1)

    def __init__(self, n: int, alpha: float = 0.05, kappa_cal: float = 0.5, warmup: int = 5, k: int = 3, m: int = 5,
                 margin_mult: float = 1.5, min_hist: int = 5, seed: int = 0):
        self.n, self.alpha, self.kc, self.warmup = n, alpha, kappa_cal, warmup
        self.k, self.m, self.margin_mult, self.min_hist = k, m, margin_mult, min_hist
        self.rng = np.random.default_rng(seed + 991)
        self.logE = {"peer_ctm": np.zeros(n), "self_ctm": np.zeros(n), "peer_bet": np.zeros(n), "self_bet": np.zeros(n), "raw_ctm": np.zeros(n), "raw_bet": np.zeros(n)}
        self.lam = 0.5   # bounded bet: factor 1 + lam (1 - 2p) lies in [1 - lam, 1 + lam]
        self.rank_hist = [[] for _ in range(n)]
        self.offences = [[] for _ in range(n)]
        self.E_RULES = ("peer_ctm", "self_ctm", "peer_bet", "self_bet", "raw_ctm", "raw_bet")
        self.slashed_at = {r: np.full(n, -1) for r in ("margin", "kofm")}
        for r_ in self.E_RULES:
            for a in self.ALPHAS:
                self.slashed_at[f"{r_}@{a}"] = np.full(n, -1)
        self.raw_hist = [[] for _ in range(n)]
        self.thr = np.log(1.0 / alpha)
        self.trace = None            # set to [] to record per-round scores and p-values (diagnostics)

    def _calib(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-12, 1.0)
        return np.log(self.kc) + (self.kc - 1.0) * np.log(p)

    def _conformal_peer(self, s: np.ndarray) -> np.ndarray:
        """p_i = (#{j: s_j > s_i} + U * #{j: s_j == s_i}) / n  (smoothed, randomised; small p = unusually bad)."""
        n = len(s); p = np.empty(n)
        for i in range(n):
            gt = np.sum(s > s[i]); eq = np.sum(s == s[i])
            p[i] = (gt + self.rng.random() * eq) / n
        return p

    def update(self, r: int, U: torch.Tensor, defense):
        s = badness_scores(U, defense)
        n = len(s)
        rank = (np.argsort(np.argsort(s)) + 0.5) / n                   # 0 = best, 1 = worst this round
        last = getattr(defense, "last", None) or {}
        thr = last.get("thr", None)
        offence = (s > (thr * self.margin_mult if thr not in (None, float("inf")) and np.isfinite(thr) else np.quantile(s, 0.9) * self.margin_mult))
        if r < self.warmup:                                              # no slashing while the model is still forming
            for i in range(n):
                self.rank_hist[i].append(rank[i]); self.raw_hist[i].append(s[i])
            return
        # own ABSOLUTE score vs own past (no peer ranks, so other tenants cannot shift it)
        p_raw = np.ones(n)
        for i in range(n):
            h = np.asarray(self.raw_hist[i])
            if len(h) >= self.min_hist:
                gt = np.sum(h > s[i]); eq = np.sum(h == s[i])
                p_raw[i] = (gt + self.rng.random() * (eq + 1)) / (len(h) + 1)
            self.raw_hist[i].append(s[i])
        self.logE["raw_ctm"] += self._calib(p_raw)
        self.logE["raw_bet"] += np.log1p(self.lam * (1 - 2 * p_raw))
        # naive per-round margin
        for i in np.flatnonzero(offence):
            if self.slashed_at["margin"][i] < 0:
                self.slashed_at["margin"][i] = r
        # k offences in the last m rounds
        for i in range(n):
            self.offences[i].append(bool(offence[i]))
            if self.slashed_at["kofm"][i] < 0 and sum(self.offences[i][-self.m:]) >= self.k:
                self.slashed_at["kofm"][i] = r
        # peer conformal e-process
        p_peer = self._conformal_peer(s)
        self.logE["peer_ctm"] += self._calib(p_peer)
        # self-referential conformal e-process on the tenant's own rank history
        p_self = np.ones(n)
        for i in range(n):
            h = np.asarray(self.rank_hist[i])
            if len(h) >= self.min_hist:
                gt = np.sum(h > rank[i]); eq = np.sum(h == rank[i])
                p_self[i] = (gt + self.rng.random() * (eq + 1)) / (len(h) + 1)
            self.rank_hist[i].append(rank[i])
        self.logE["self_ctm"] += self._calib(p_self)
        # bounded-bet e-processes: E[1 + lam (1 - 2p)] = 1 for uniform p, and each factor is in [0.5, 1.5]
        self.logE["peer_bet"] += np.log1p(self.lam * (1 - 2 * p_peer))
        self.logE["self_bet"] += np.log1p(self.lam * (1 - 2 * p_self))
        if self.trace is not None:
            self.trace.append(dict(r=int(r), s=np.round(s, 5).tolist(), p_peer=np.round(p_peer, 5).tolist(),
                                   p_self=np.round(p_self, 5).tolist(), p_raw=np.round(p_raw, 5).tolist()))
        for rule in self.E_RULES:
            for a in self.ALPHAS:
                key = f"{rule}@{a}"
                hit = (self.logE[rule] >= np.log(1.0 / a)) & (self.slashed_at[key] < 0)
                self.slashed_at[key][hit] = r

    def summary(self, mal: list, attack_start: int) -> dict:
        mal = set(mal); hon = [i for i in range(self.n) if i not in mal]
        out = {}
        for rule, at in self.slashed_at.items():
            out[f"acct_{rule}_honest_slashed"] = float(np.mean([at[i] >= 0 for i in hon])) if hon else float("nan")
            out[f"acct_{rule}_mal_slashed"] = float(np.mean([at[i] >= 0 for i in mal])) if mal else float("nan")
            delays = [at[i] - attack_start for i in mal if at[i] >= 0]
            out[f"acct_{rule}_rounds_to_slash"] = float(np.mean(delays)) if delays else float("nan")
        return out
