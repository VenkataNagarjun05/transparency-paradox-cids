"""Data loading, cleaning and tenant partitioning.

Sources are the NetFlow-v2 versions of four public IDS datasets. They share one
feature schema, so each source can stand in for a different cloud tenant type.
Everything here is schema-tolerant: label columns are detected by name and
non-numeric / identifier columns are dropped, because the Kaggle copies have
not been opened on the author's machine yet. The inspection notebook prints
the real schema before any experiment runs.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Kaggle slugs checked against the Kaggle public API on 2026-10-14.
KAGGLE_SOURCES = {
    "unsw": "dhoogla/nfunswnb15v2",
    "cse18": "dhoogla/nfcsecicids2018v2",
    "ton": "dhoogla/nftoniotv2",
    "bot": "dhoogla/nfbotiotv2",
}
EXTRA_SOURCES = {
    "ciciot": "himadri07/ciciot2023",
    "kube": "redamorsli/kube-ids0",
}

LABEL_COLS = ("label", "attack", "attack_cat", "class", "target")
DROP_PATTERNS = re.compile(r"(ADDR|_PORT$|^SRC_|^DST_IP|DNS_QUERY_ID|^ID$|^FLOW_ID|TIMESTAMP|^TIME)", re.I)
# Corrected identifier filter: "^SRC_" also matched the traffic features SRC_TO_DST_SECOND_BYTES and SRC_TO_DST_AVG_THROUGHPUT, while their
# DST_TO_SRC_* counterparts were kept. "^SRC_IP" removes only source-address columns. The original filter is kept as the default so that
# the R1-R3 results reproduce exactly; R4 (docs/PREREG_R4.md) re-runs the primary comparison with the corrected one.
DROP_PATTERNS_FIXED = re.compile(r"(ADDR|_PORT$|^SRC_IP|^DST_IP|DNS_QUERY_ID|^ID$|^FLOW_ID|TIMESTAMP|^TIME)", re.I)
_FILTER = {"original": DROP_PATTERNS, "fixed": DROP_PATTERNS_FIXED}

FAMILIES = ["benign", "dos", "recon", "bruteforce", "web", "exploit", "persist", "other"]
_FAMILY_RULES = [
    ("benign", r"^(benign|normal|0)$"),
    ("dos", r"ddos|dos|slowloris|hulk|goldeneye|slowhttp|loic|hoic"),
    ("recon", r"scan|reconnaissance|recon|probe"),
    ("bruteforce", r"brute|password|ssh|ftp"),
    ("web", r"injection|xss|sql|web"),
    ("exploit", r"exploit|shellcode|generic|analysis|worm|fuzz|ransom"),
    ("persist", r"backdoor|bot|infil|theft|mitm"),
]


def family_of(name) -> str:
    s = str(name).strip().lower()
    for fam, pat in _FAMILY_RULES:
        if re.search(pat, s):
            return fam
    return "other"


def download(key: str) -> str:
    """Download a Kaggle dataset with kagglehub and return its local path."""
    import kagglehub

    slug = {**KAGGLE_SOURCES, **EXTRA_SOURCES}[key]
    return kagglehub.dataset_download(slug)


def _files(path: str):
    out = []
    for ext in ("parquet", "csv"):
        out += glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
    return sorted(out)


def _has_label(path: str) -> bool:
    """True if the file is a data file (has a Label / Attack column), not a documentation file."""
    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            cols = pq.ParquetFile(path).schema_arrow.names
        else:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return False
    return any(c.lower() in LABEL_COLS for c in cols)


def _detect_labels(cols):
    low = {c.lower(): c for c in cols}
    lab = next((low[c] for c in ("label",) if c in low), None)
    att = next((low[c] for c in ("attack", "attack_cat", "class") if c in low), None)
    return lab, att


def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def _keep_col(c: str, pat=None) -> bool:
    """Columns worth reading at all: labels, plus anything that is not an address/port/id column."""
    return c.lower() in LABEL_COLS or not (pat or DROP_PATTERNS).search(c)


def _iter_batches(files, batch_rows=100_000, pat=None):
    """Yield small pandas frames (only the needed columns) so memory stays flat."""
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(f)
            cols = [c for c in pf.schema_arrow.names if _keep_col(c, pat)]
            for b in pf.iter_batches(batch_size=batch_rows, columns=cols, use_threads=False):
                yield b.to_pandas()
        else:
            for ch in pd.read_csv(f, chunksize=batch_rows, low_memory=False,
                                  usecols=lambda c: _keep_col(c, pat)):
                yield ch


def load_source(path: str, max_rows: int = 400_000, seed: int = 0, per_family_cap: int = 80_000,
                batch_rows: int = 100_000, verbose: bool = True, dedupe: bool = True, id_filter: str = "original"):
    """Stream a dataset with flat memory; keep a family-balanced sample; clean and dedupe.
    id_filter: "original" (as used for R1-R3) or "fixed" (keeps SRC_TO_DST_* traffic features; see DROP_PATTERNS_FIXED).

    Each batch is turned into float32 arrays and only a thin random slice of every family is kept, so
    memory is set by the sample (about per_family_cap x families rows), not by the file size.
    Returns X (float32), y (0/1), fam (str array), feature names, and a report dict.
    """
    import gc

    rng = np.random.default_rng(seed)
    all_files = _files(path)
    files = [f for f in all_files if _has_label(f)]
    skipped = [os.path.basename(f) for f in all_files if f not in files]
    if not files:
        raise FileNotFoundError(f"no data file with a Label/Attack column under {path} (saw {skipped})")
    n_total = 0
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            n_total += pq.ParquetFile(f).metadata.num_rows
    n_batches = max(1, int(np.ceil(n_total / batch_rows))) if n_total else 50
    # keep ~2x the final cap per family across all batches, then trim at the end
    per_batch_cap = max(20, int(np.ceil(2.0 * per_family_cap / n_batches)))
    rep = {"files": [os.path.basename(f) for f in files], "skipped_non_data_files": skipped,
           "rows_seen": 0, "n_batches": n_batches}
    pat = _FILTER[id_filter]
    rep["id_filter"] = id_filter
    feat, Xs, fams = None, [], []
    for i, df in enumerate(_iter_batches(files, batch_rows, pat)):
        rep["rows_seen"] += len(df)
        lab, att = _detect_labels(df.columns)
        if lab is None and att is None:
            raise ValueError(f"no label column in {df.columns.tolist()[:10]}")
        if att is not None:
            names = df[att].astype(str)
            uniq = names.unique()
            m = {u: family_of(u) for u in uniq}
            fam = names.map(m).to_numpy(dtype=object)
        else:
            fam = np.where(df[lab].to_numpy() == 0, "benign", "other").astype(object)
        if feat is None:
            feat = [c for c in df.columns if c.lower() not in LABEL_COLS and not pat.search(c)
                    and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feat].to_numpy(dtype=np.float32, copy=False)
        del df
        keep = []
        for f_ in np.unique(fam):
            ii = np.flatnonzero(fam == f_)
            if len(ii) > per_batch_cap:
                ii = rng.choice(ii, per_batch_cap, replace=False)
            keep.append(ii)
        keep = np.concatenate(keep)
        Xs.append(X[keep].copy())
        fams.append(fam[keep])
        del X, fam
        if verbose and (i % 25 == 0):
            print(f"   batch {i + 1}/{n_batches}  kept so far {sum(len(a) for a in Xs):,} rows  rss {_rss_gb():.1f} GB", flush=True)
            gc.collect()
    X = np.concatenate(Xs); fam = np.concatenate(fams)
    del Xs, fams
    gc.collect()
    ok = np.isfinite(X).all(axis=1)
    X, fam = X[ok], fam[ok]
    n0 = len(X)
    key = pd.util.hash_pandas_object(
        pd.concat([pd.DataFrame(X), pd.Series(fam, name="fam")], axis=1), index=False).to_numpy()
    _, first = np.unique(key, return_index=True)
    first = np.sort(first) if dedupe else np.arange(len(key))   # dedupe=False keeps exact duplicates (inflation experiment)
    X, fam = X[first], fam[first]
    rep.update(rows_after_nan=int(n0), rows_after_dedupe=int(len(X)),
               dup_frac=float(1 - len(X) / max(n0, 1)), n_features=len(feat))
    # constant columns carry no signal
    nz = X.std(axis=0) > 0
    X = X[:, nz]
    rep["constant_features_removed"] = [c for c, k in zip(feat, nz) if not k]
    feat = [c for c, k in zip(feat, nz) if k]
    rep["n_features_final"] = len(feat)
    # family-balanced cap
    keep = np.zeros(len(X), dtype=bool)
    for f_ in np.unique(fam):
        ii = np.flatnonzero(fam == f_)
        if len(ii) > per_family_cap:
            ii = rng.choice(ii, per_family_cap, replace=False)
        keep[ii] = True
    X, fam = X[keep], fam[keep]
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, fam = X[idx], fam[idx]
    y = (fam != "benign").astype(np.int64)
    rep["family_counts"] = {str(k): int(v) for k, v in zip(*np.unique(fam, return_counts=True))}
    rep["rows_final"] = int(len(X))
    return X.astype(np.float32), y, fam.astype(str), feat, rep


def squash(X: np.ndarray) -> np.ndarray:
    """Sign-preserving log transform for heavy-tailed flow features."""
    return np.sign(X) * np.log1p(np.abs(X))


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X):
        Z = squash(X)
        return cls(Z.mean(0), Z.std(0) + 1e-6)

    def __call__(self, X):
        return np.clip((squash(X) - self.mean) / self.std, -5, 5).astype(np.float32)


@dataclass
class World:
    """Everything one experiment needs."""
    n_features: int
    tenants: list          # list of dict(X, y, fam, source)
    test: dict             # source -> dict(X, y, fam)
    audit: dict            # source -> dict(X, y, fam)  pool for probes (disjoint from train/test)
    sources: list
    meta: dict = field(default_factory=dict)


def build_world(pools: dict, n_tenants: int = 20, n_train: int = 3000, n_test: int = 6000,
                n_audit: int = 8000, alpha: float = 1.0, seed: int = 0,
                holdout_source: str | None = None, scaler_fit: str = "train") -> World:
    """pools: source -> (X, y, fam) of RAW features. Splits into tenants / test / audit.

    Tenants are spread evenly over the training sources. Within a source the class
    mix of each tenant is drawn from Dirichlet(alpha * global mix) for label skew.
    """
    rng = np.random.default_rng(seed)
    sources = [s for s in pools if s != holdout_source]
    # Split first (raw indices), then fit the scaler on the tenants' TRAINING rows only and freeze it for the probe pool
    # and the test sets. Fitting on the union of tenant rows equals aggregating per-tenant feature means and variances,
    # which a consortium can do without sharing flows. (scaler_fit="all" reproduces the earlier, leaky behaviour.)
    test_idx, audit_idx, left = {}, {}, {}
    for s, (X, y, fam) in pools.items():
        idx = rng.permutation(len(X))
        nt = min(n_test, len(X) // 4)
        na = min(n_audit, len(X) // 4)
        test_idx[s], audit_idx[s], left[s] = idx[:nt], idx[nt:nt + na], idx[nt + na:]
    per = max(1, n_tenants // len(sources))
    tenant_sel = []
    for s in sources:
        X, y, fam = pools[s]
        rest = left[s]
        fams = np.unique(fam[rest])
        base = np.array([np.mean(fam[rest] == f) for f in fams])
        base = np.maximum(base, 0.02)
        base /= base.sum()
        by = {f: rng.permutation(rest[fam[rest] == f]) for f in fams}
        ptr = {f: 0 for f in fams}
        for _ in range(per):
            p = rng.dirichlet(alpha * len(fams) * base + 0.05)
            cnt = np.maximum(1, np.round(p * n_train).astype(int))
            sel = []
            for f, c in zip(fams, cnt):
                pool_f = by[f]
                take = pool_f[np.arange(ptr[f], ptr[f] + c) % len(pool_f)]
                ptr[f] += c
                sel.append(take)
            sel = np.concatenate(sel)
            sel = rng.permutation(sel)[:n_train]
            if len(sel) < n_train:  # pad by resampling
                sel = np.concatenate([sel, rng.choice(sel, n_train - len(sel))])
            tenant_sel.append((s, sel))
    if scaler_fit == "all":
        scaler = Scaler.fit(np.concatenate([pools[s][0] for s in sources]))
    else:
        train_rows = np.concatenate([pools[s][0][np.unique(sel)] for s, sel in tenant_sel])
        scaler = Scaler.fit(train_rows)
    tenants = [dict(X=scaler(pools[s][0][sel]), y=pools[s][1][sel], fam=pools[s][2][sel], source=s) for s, sel in tenant_sel]
    test, audit = {}, {}
    for s, (X, y, fam) in pools.items():
        test[s] = dict(X=scaler(X[test_idx[s]]), y=y[test_idx[s]], fam=fam[test_idx[s]])
        if s != holdout_source:          # a held-out source must not leak into the probe pool
            audit[s] = dict(X=scaler(X[audit_idx[s]]), y=y[audit_idx[s]], fam=fam[audit_idx[s]])
    return World(n_features=pools[sources[0]][0].shape[1], tenants=tenants, test=test, audit=audit, sources=sources,
                 meta=dict(alpha=alpha, n_tenants=len(tenants), n_train=n_train, holdout=holdout_source, scaler_fit=scaler_fit))


def split_pools(pools: dict, seed: int = 12345):
    """Split every source into two DISJOINT halves, stratified by family: one only for tuning
    hyper-parameters, one only for the final evaluation. Rows never appear in both."""
    rng = np.random.default_rng(seed)
    tune, final = {}, {}
    for s, (X, y, fam) in pools.items():
        a = np.zeros(len(X), dtype=bool)
        for f in np.unique(fam):
            ii = rng.permutation(np.flatnonzero(fam == f))
            a[ii[: len(ii) // 2]] = True
        tune[s] = (X[a], y[a], fam[a])
        final[s] = (X[~a], y[~a], fam[~a])
    return tune, final


KUBE_LABELS = {0: "benign", 1: "dos", 2: "dos", 3: "bruteforce", 4: "web"}   # per the dataset README (DVWA / Bank of Anthos)


def load_kube(path: str, max_rows: int = 200_000, seed: int = 0, verbose: bool = True):
    """Kubernetes cloud-native IDS data (redamorsli/kube-ids0): the two ML-ready files, one per application.
    Returns pools {'kube_dvwa': (X, y, fam), 'kube_boa': (...)} on the SHARED numeric columns, plus a report."""
    rng = np.random.default_rng(seed)
    files = {"kube_dvwa": glob.glob(os.path.join(path, "**", "dvwa_dataset_ml_ready*.csv"), recursive=True),
             "kube_boa": glob.glob(os.path.join(path, "**", "boa_dataset_ml_ready*.csv"), recursive=True)}
    frames, rep = {}, {}
    for k, fs in files.items():
        if not fs:
            continue
        df = pd.read_csv(fs[0], low_memory=False)
        lab = next((c for c in df.columns if "label" in c.lower() or c.lower() in ("attack", "class", "target")), None)
        rep[k] = {"file": os.path.basename(fs[0]), "rows": len(df), "label_col": lab, "columns": df.columns.tolist()[:60]}
        if lab is None:
            continue
        v = df[lab]
        fam = v.map(lambda x: KUBE_LABELS.get(int(x), "other")) if pd.api.types.is_numeric_dtype(v) else v.map(family_of)
        df = df.assign(__fam=fam.values)
        frames[k] = df
        if verbose:
            print(k, os.path.basename(fs[0]), df.shape, "label:", lab, "families:", df["__fam"].value_counts().to_dict())
    if not frames:
        raise FileNotFoundError("no ml_ready kube files with a label column")
    num = [set(c for c in f.columns if c != "__fam" and c.lower() not in LABEL_COLS and "label" not in c.lower()
               and not DROP_PATTERNS.search(c) and not re.search(r"(ip|time|stamp|id$)", c, re.I)
               and pd.api.types.is_numeric_dtype(f[c])) for f in frames.values()]
    common = sorted(set.intersection(*num)) if len(num) > 1 else sorted(num[0])
    if len(common) < 8 and len(frames) > 1:                     # the two apps share too little: keep DVWA only
        frames = {"kube_dvwa": frames["kube_dvwa"]} if "kube_dvwa" in frames else dict(list(frames.items())[:1])
        common = sorted(num[0])
    rep["common_features"] = common
    pools = {}
    for k, f in frames.items():
        X = f[common].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
        fam = f["__fam"].to_numpy().astype(str)
        key = pd.util.hash_pandas_object(pd.concat([pd.DataFrame(X), pd.Series(fam)], axis=1), index=False).to_numpy()
        _, first = np.unique(key, return_index=True); first = np.sort(first)
        rep[k]["dup_frac"] = float(1 - len(first) / max(len(X), 1))
        X, fam = X[first], fam[first]
        if len(X) > max_rows:
            idx = rng.choice(len(X), max_rows, replace=False); X, fam = X[idx], fam[idx]
        nz = X.std(0) > 0
        pools[k] = (X, (fam != "benign").astype(np.int64), fam)
        rep[k]["rows_final"] = int(len(X))
    keep = np.ones(len(common), dtype=bool)
    for X, _, _ in pools.values():
        keep &= X.std(0) > 0
    pools = {k: (X[:, keep], y, f) for k, (X, y, f) in pools.items()}
    rep["n_features"] = int(keep.sum())
    return pools, rep


def synthetic_pools(n_features=43, n_sources=4, n_per=30000, seed=0, sep=0.22):
    """Stand-in data for local smoke tests only. Never used for reported results."""
    rng = np.random.default_rng(seed)
    fams = ["benign", "dos", "recon", "bruteforce", "web", "exploit"]
    pools = {}
    centers = rng.normal(0, sep, size=(len(fams), n_features))
    for k in range(n_sources):
        shift = rng.normal(0, 0.15, size=n_features)
        mix = np.array([0.5, 0.2, 0.1, 0.07, 0.07, 0.06])
        fam = rng.choice(fams, size=n_per, p=mix)
        ci = np.array([fams.index(f) for f in fam])
        X = centers[ci] + shift + rng.normal(0, 1.0, size=(n_per, n_features))
        X = np.sign(X) * (np.expm1(np.abs(X) * 1.5))  # heavy tails like raw flow counters
        y = (fam != "benign").astype(np.int64)
        pools[f"src{k}"] = (X.astype(np.float32), y, fam.astype(str))
    return pools


def report_json(reports: dict, path: str):
    with open(path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
