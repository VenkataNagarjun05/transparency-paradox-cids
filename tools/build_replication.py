"""Replication notebooks R1 and R2 (pre-registered in docs/PREREG_REPLICATION.md).

Why: an external review found that the feature scaler was fitted on pooled flows that included test rows. The scaler is now fitted
on tenant training rows only (qbfl/data.py, scaler_fit="train"), so every study is re-run, on NEW seeds, with the analysis fixed in
advance. Both notebooks embed the CURRENT qbfl sources (they do not copy cells from older notebooks).

R1: tuning (half A) -> main study -> ablation -> memorizing attacker -> confirmatory sealing test        (~6 GPU h)
R2: tuning (identical, re-derived) -> optimising attacker (+4x budget, +published min-max with full knowledge) -> accountability with
    p-value traces -> detection quality -> leave-one-source-out with diagnostics -> reconnaissance -> sensitivity -> Kubernetes ->
    energy -> software versions                                                                          (~5.5 GPU h)
"""
import json, pathlib, uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = {p.name: p.read_text() for p in sorted((ROOT / "qbfl").glob("*.py"))}

SEEDS = dict(TUNE=(200, 201, 202), MAIN=tuple(range(300, 310)), MEM=tuple(range(310, 320)), SETTLE=tuple(range(320, 350)),
             STRONG=tuple(range(350, 360)), LOSO=tuple(range(360, 365)), RECON=tuple(range(365, 370)), ACCT=tuple(range(370, 380)),
             SENS=tuple(range(380, 385)), DETECT=tuple(range(390, 395)), KUBE=tuple(range(400, 405)), ENERGY=500)


def md(t): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": t.splitlines(True)}
def code(t, done):
    t = t.rstrip("\n") + f"\nprint('\\n\u2714 CELL COMPLETED: {done}')"
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}
def guarded(body, name):
    ind = "\n".join("    " + l for l in body.strip("\n").splitlines())
    return f"try:\n{ind}\nexcept Exception as e:\n    import traceback; traceback.print_exc(); FAILED.append('{name}'); print('STAGE FAILED (continuing):', '{name}', repr(e))"
def stage(name, body, done):
    return code(f"if RUN.get('{name}', False):\n" + "\n".join("    " + l for l in guarded(body, name).splitlines()) + f"\nelse:\n    print('SKIPPED: {name}')", done)


WRITE = ("# Writes the CURRENT qbfl package next to this notebook (scaler fitted on tenant training rows only).\n"
         "import pathlib, json\n"
         f"FILES = json.loads({json.dumps(json.dumps(PKG))})\n"
         "pathlib.Path('qbfl').mkdir(exist_ok=True)\n"
         "for name, src in FILES.items():\n    pathlib.Path('qbfl', name).write_text(src)\n"
         "print('wrote', sorted(FILES))")

COMMON = [
    code("!pip -q install kagglehub pyarrow cryptography psutil\ntry:\n    import subprocess, sys\n"
         "    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'liboqs-python'], check=True, timeout=1500)\n"
         "except Exception as e:\n    print('liboqs-python install failed; the post-quantum cell will be skipped:', e)", "packages installed"),
    code(WRITE, "qbfl package written to disk"),
    code("import sys, os, json, glob, time, platform, numpy as np, torch, pandas as pd\nsys.path.insert(0, '.')\n"
         "from qbfl import data as D, grid, analysis as A\nfrom qbfl.fl import Config, run_one\n"
         "pd.set_option('display.width', 230, 'display.max_rows', 400, 'display.max_columns', 40)\n"
         "GPUS = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']\nos.makedirs('results', exist_ok=True)\n"
         "import sklearn, scipy, pyarrow\nVERS = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,\n"
         "            pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, pyarrow=pyarrow.__version__,\n"
         "            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
         "try:\n    import oqs; VERS['liboqs_python'] = getattr(oqs, '__version__', '?'); VERS['liboqs'] = oqs.oqs_version()\nexcept Exception as e:\n    VERS['liboqs'] = repr(e)\n"
         "json.dump(VERS, open('results/versions.json', 'w'), indent=1); print(json.dumps(VERS, indent=1))",
         "libraries imported, versions recorded"),
    code("tp, fp_ = D.split_pools(D.synthetic_pools(n_per=12000))\nw_s = D.build_world(tp, n_tenants=12, n_train=500, n_test=1200, n_audit=2500)\n"
         "for d in grid.FINAL_DEFENSES:\n    for a in ('rule_aware', 'opt_adaptive_x4'):\n"
         "        r = run_one(w_s, Config(defense=d, attack=a, mal_frac=0.25, rounds=4, eval_every=2, local_epochs=1, attack_start=2, dparams=grid.TUNE_GRIDS[d][0]), GPUS[0])\n"
         "    print(f\"{d:13s} ok\")\nprint('SELF-TEST PASSED; scaler_fit =', w_s.meta['scaler_fit'])", "self-test passed"),
    code("import kagglehub\nSOURCES = list(D.KAGGLE_SOURCES)\npaths = {k: D.download(k) for k in SOURCES}\nprint(paths)", "datasets downloaded"),
    code("REPORT, raw = {}, {}\nfor k in SOURCES:\n    t = time.time()\n    X, y, fam, feat, rep = D.load_source(paths[k], max_rows=400_000, per_family_cap=60_000, seed=0, batch_rows=100_000, verbose=False)\n"
         "    raw[k] = (X, y, fam, feat); REPORT[k] = rep\n    print(f\"{k}: {len(X):,} rows, dup removed {rep['dup_frac']:.1%}, {time.time()-t:.0f}s\", rep['family_counts'])\n"
         "common = [c for c in raw[SOURCES[0]][3] if all(c in raw[k][3] for k in SOURCES)]\npools = {k: (raw[k][0][:, [raw[k][3].index(c) for c in common]], raw[k][1], raw[k][2]) for k in SOURCES}\n"
         "REPORT['common_features'] = common; D.report_json(REPORT, 'results/data_report.json')\nassert all('dos' in set(pools[k][2]) for k in SOURCES)\nprint(len(common), 'shared features. DATA OK')",
         "data cleaned, deduplicated and aligned"),
    code("SEEDS = json.loads('" + json.dumps(SEEDS) + "')\nSEEDS = {k: (tuple(v) if isinstance(v, list) else v) for k, v in SEEDS.items()}\nFAILED = []\n"
         "pools_tune, pools_final = D.split_pools(pools, seed=12345)\n"
         "def world_from(p, **kw):\n    a = dict(n_tenants=20, n_train=3000, n_test=6000, n_audit=9000, alpha=1.0, seed=0, scaler_fit='train'); a.update(kw)\n    return D.build_world(p, **a)\n"
         "world_tune = world_from(pools_tune, seed=1)\nworld_final = world_from(pools_final)\n"
         "def go(world, cfgs, name):\n    out = f'results/{name}'\n    print(name, len(cfgs), 'configs on', GPUS, flush=True)\n"
         "    if len(GPUS) > 1: grid.run_parallel(world, cfgs, out, GPUS, tag=name)\n    else: grid.run_grid(world, cfgs, out + '.shard0.jsonl', GPUS[0], tag=name)\n"
         "print('worlds built with scaler_fit =', world_final.meta['scaler_fit'], '|', len(world_final.tenants), 'tenants,', world_final.n_features, 'features')",
         "worlds built (scaler fitted on tenant training rows only)"),
    md("## Tuning on half A (identical procedure and grids as before; seeds 200-202)"),
    code(guarded("go(world_tune, grid.tune_preset(seeds=SEEDS['TUNE']), 'tune')\ndft = A.load('results/tune*.jsonl')\nBEST, T = A.select_best(dft)\n"
                 "json.dump(BEST, open('results/best_params.json', 'w'), indent=1, sort_keys=True); T.to_csv('results/tuning_table.csv', index=False)\n"
                 "print(len(dft), 'tuning runs'); print(json.dumps(BEST, indent=1, sort_keys=True))", "TUNE"), "tuning finished, settings selected"),
]

# ------------------------------------------------------------------ R1 stages
R1 = {
 "MAIN": "go(world_final, grid.final_preset(BEST, seeds=SEEDS['MAIN']), 'final')",
 "ABLATION": "go(world_final, grid.ablation_preset(BEST, seeds=SEEDS['MAIN']), 'ablation')",
 "MEM": "go(world_final, grid.memorize_preset(BEST, seeds=SEEDS['MEM']), 'memorize')",
 "SETTLE": "go(world_final, grid.settle_preset(BEST, seeds=SEEDS['SETTLE']), 'settle')",
 "PQ": ("from qbfl import pqc\nprint(json.dumps(pqc.security_checks(), indent=1)); rows = []\n"
        "for sig, kem in [('ML-DSA-44','ML-KEM-512'), ('ML-DSA-65','ML-KEM-768'), ('ML-DSA-87','ML-KEM-1024')]:\n"
        "    for n_p, n_t in [(45_000, 20), (45_000, 100), (1_000_000, 20)]:\n        b = pqc.bench(n_p, n_t, sig, kem, reps=50); b.update(sig=sig, kem=kem); rows.append(b)\n"
        "pd.DataFrame(rows).to_csv('results/pq_costs.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())"),
}
R1_ANALYSIS = r'''
from scipy.stats import wilcoxon
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj
def paired(d, a, b, atk, frac):
    g = d[(d.attack == atk) & (d.mal_frac == frac)]
    pa = g[g.defense == a].set_index('seed').final_rec_target; pb = g[g.defense == b].set_index('seed').final_rec_target
    i = pa.index.intersection(pb.index); diff = (pa[i] - pb[i]).values
    try: p = float(wilcoxon(diff, alternative='greater').pvalue)
    except ValueError: p = 1.0
    bs = np.random.default_rng(0).choice(diff, (10000, len(diff))).mean(1)
    return dict(n=len(diff), diff=diff.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p)
d = A.load('results/final*.jsonl'); print(len(d), 'main runs')
print(d[(d.mal_frac == 0.3) | (d.attack == 'none')].pivot_table(index='defense', columns='attack', values='final_rec_target', aggfunc='mean').round(3))
s = A.load('results/settle*.jsonl'); s = s[s.leak_frac == 0]; print(len(s), 'settle runs')
tests = [('sealed_fam', 'public_fam'), ('sealed_fam', 'probe_public'), ('sealed_abs', 'probe_public')]
res = [paired(s, a, b, 'probe_memorize', 0.3) for a, b in tests]; adj = holm(np.array([r['p'] for r in res]))
for (a, b), r, q in zip(tests, res, adj): print(f"H2 {a} - {b}: {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] p={r['p']:.4f} Holm={q:.4f}")
'''

# ------------------------------------------------------------------ R2 stages
R2 = {
 "STRONG": ("go(world_final, grid.strong_preset(BEST, seeds=SEEDS['STRONG']), 'strong')\n"
            "go(world_final, grid.p2_preset(BEST, grid.STRONG_DEFENSES, ['opt_adaptive_x4'], SEEDS['STRONG'], fracs=(0.3,)), 'strong_x4')\n"
            "go(world_final, grid.p2_preset(BEST, grid.STRONG_DEFENSES, ['minmax'], SEEDS['STRONG'], fracs=(0.3,), see_benign=True), 'strong_minmax_fk')"),
 "ACCT": "go(world_final, grid.p2_preset(BEST, grid.ACCT_DEFENSES, grid.ACCT_ATTACKS, SEEDS['ACCT'], extra={'acct_trace': True}), 'acct')",
 "DETECT": r'''
from qbfl.fl import run_reference
ref_rows = []
for hid in [(128, 64), (256, 128, 64)]:
    for s in SEEDS['DETECT']:
        for mode in ('centralized', 'local_only'):
            r = run_reference(world_final, Config(seed=s, hidden=hid), GPUS[0], mode=mode, epochs=20); r['data'] = 'dedup'; ref_rows.append(r)
go(world_final, grid.p2_preset(BEST, ['fedavg', 'median', 'cosrep', 'sealed_fam'], ['none'], SEEDS['DETECT'], hidden=(256, 128, 64), rounds=60), 'detect_strong')
go(world_final, grid.p2_preset(BEST, ['fedavg', 'median', 'cosrep', 'sealed_fam'], ['none'], SEEDS['DETECT']), 'detect_base')
raw_dup = {}
for k in SOURCES:
    X, y, fam, feat, rep = D.load_source(paths[k], max_rows=400_000, per_family_cap=60_000, seed=0, batch_rows=100_000, dedupe=False, verbose=False)
    raw_dup[k] = (X[:, [feat.index(c) for c in common]], y, fam)
_, pools_final_dup = D.split_pools(raw_dup, seed=12345)
world_dup = world_from(pools_final_dup)
for s in SEEDS['DETECT']:
    r = run_reference(world_dup, Config(seed=s), GPUS[0], mode='centralized', epochs=20); r['data'] = 'dup_kept'; ref_rows.append(r)
go(world_dup, grid.p2_preset(BEST, ['fedavg', 'sealed_fam'], ['none'], SEEDS['DETECT']), 'detect_dupkept')
with open('results/detect_reference.jsonl', 'w') as f:
    for r in ref_rows: f.write(json.dumps(r, default=float) + '\n')
''',
 "LOSO": r'''
from qbfl import model as M
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
diag = {}
for hs in SOURCES:
    w = world_from(pools_final, holdout_source=hs)
    go(w, grid.p2_preset(BEST, grid.P2_DEFENSES, ['none', 'rule_aware', 'opt_adaptive'], SEEDS['LOSO']), f'loso_{hs}')
    # diagnostics: label mix, feature shift, per-family recall of a clean FedAvg model on the held-out source
    Xtr = np.concatenate([t['X'] for t in w.tenants]); Ftr = np.concatenate([t['fam'] for t in w.tenants]); te = w.test[hs]
    smd = np.abs(te['X'].mean(0) - Xtr.mean(0)) / (Xtr.std(0) + 1e-6)
    ks = [ks_2samp(te['X'][:, j], Xtr[:, j]).statistic for j in range(Xtr.shape[1])]
    fams_tr = {f: float(np.mean(Ftr == f)) for f in np.unique(Ftr)}; fams_te = {f: float(np.mean(te['fam'] == f)) for f in np.unique(te['fam'])}
    r = run_one(w, Config(defense='fedavg', attack='none', seed=SEEDS['LOSO'][0]), GPUS[0])
    diag[hs] = dict(train_family_mix=fams_tr, heldout_family_mix=fams_te, families_unseen_in_training=sorted(set(fams_te) - set(fams_tr)),
                    smd_median=float(np.median(smd)), smd_frac_gt1=float(np.mean(smd > 1)), ks_median=float(np.median(ks)),
                    top_shift_features=[common[j] for j in np.argsort(-smd)[:5]],
                    heldout_metrics={k: v for k, v in r['final'].items() if k.endswith('_' + hs)},
                    heldout_family_recall={k: v for k, v in r['final'].items() if k.startswith('rec_') and k.endswith('_' + hs) and not k.startswith('rec_target')})
    print(hs, json.dumps({k: v for k, v in diag[hs].items() if k not in ('train_family_mix',)}, default=float)[:900])
json.dump(diag, open('results/loso_diag.json', 'w'), indent=1, default=float)
''',
 "RECON": "go(world_final, grid.p2_preset(BEST, grid.P2_DEFENSES, ['none', 'rule_aware', 'probe_memorize', 'opt_adaptive'], SEEDS['RECON'], target='recon'), 'recon')",
 "SENS": ("for name, kw in [('sens_alpha0.3', dict(alpha=0.3)), ('sens_alpha10', dict(alpha=10.0)), ('sens_50tenants', dict(n_tenants=50, n_train=1200))]:\n"
          "    w = world_from(pools_final, **kw); go(w, grid.p2_preset(BEST, grid.P2_DEFENSES, ['none', 'rule_aware', 'opt_adaptive'], SEEDS['SENS']), name)"),
 "KUBE": ("kpath = D.download('kube'); kpools, krep = D.load_kube(kpath, max_rows=200_000, seed=0); D.report_json(krep, 'results/kube_report.json')\n"
          "assert not any(('port' in c.lower()) or c.lower().endswith('ip') for c in krep['common_features']), 'label-defining port column leaked'\n"
          "kt, kf = D.split_pools(kpools, seed=12345)\nwk = D.build_world(kf, n_tenants=10, n_train=1500, n_test=4000, n_audit=4000, alpha=1.0, seed=0, scaler_fit='train')\n"
          "go(wk, grid.p2_preset(BEST, grid.P2_DEFENSES, ['none', 'rule_aware', 'probe_memorize', 'opt_adaptive'], SEEDS['KUBE']), 'kube')"),
 "ENERGY": r'''
import subprocess, threading
def gpu_power_w():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', '0'], capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return float('nan')
def measure(cfg):
    samples, stop = [], [False]
    def loop():
        while not stop[0]:
            samples.append(gpu_power_w()); time.sleep(0.25)
    th = threading.Thread(target=loop); th.start(); t0 = time.time()
    r = run_one(world_final, cfg, GPUS[0]); stop[0] = True; th.join(); dt = time.time() - t0
    p = np.array(samples); e = float(np.nanmean(p) * dt) if len(p) else float('nan')
    return dict(defense=cfg.defense, attack=cfg.attack, rounds=cfg.rounds, wall_s=dt, sec_per_round=dt / cfg.rounds, mean_power_w=float(np.nanmean(p)),
                energy_j=e, energy_j_per_round=e / cfg.rounds, defense_sec_per_round=r['defense_sec'] / cfg.rounds)
rows = [measure(Config(defense=dn, attack=at, mal_frac=0.3, seed=SEEDS['ENERGY'], dparams=BEST.get(dn, {})))
        for dn, at in [('fedavg', 'none'), ('median', 'none'), ('cosrep', 'none'), ('probe_public', 'none'), ('sealed_fam', 'none'),
                       ('sealed_fam', 'probe_memorize'), ('sealed_fam', 'opt_adaptive'), ('cosrep', 'opt_adaptive'), ('sealed_fam', 'opt_adaptive_x4')]]
pd.DataFrame(rows).round(4).to_csv('results/energy_latency.csv', index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))
''',
}
R2_ANALYSIS = r'''
from scipy.stats import wilcoxon
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj
d = A.load('results/strong.shard*.jsonl'); print(len(d), 'strong runs')
rows = []
for dn in sorted(d.defense.unique()):
    c = d[(d.defense == dn) & (d.attack == 'none')].set_index('seed').final_rec_target
    o = d[(d.defense == dn) & (d.attack == 'opt_adaptive') & (d.mal_frac == 0.3)].set_index('seed').final_rec_target
    i = c.index.intersection(o.index); diff = (c[i] - o[i]).values
    try: p = float(wilcoxon(diff, alternative='greater').pvalue)
    except ValueError: p = 1.0
    rows.append((dn, len(i), diff.mean(), (diff > 0).mean(), p))
adj = holm(np.array([r[4] for r in rows]))
print('H1 (pre-registered): recall drop under the optimiser at 30%, one-sided Wilcoxon, Holm over all rules')
for (dn, n, m, frac, p), q in zip(rows, adj): print(f'  {dn:13s} n={n} drop={m:+.3f} seeds-with-drop={frac:.0%} p={p:.4f} Holm={q:.4f}')
for f in ('strong_x4', 'strong_minmax_fk'):
    g = A.load(f'results/{f}*.jsonl'); print(f, len(g)); print(g.groupby('defense').final_rec_target.mean().round(3).to_string())
'''

# ------------------------------------------------------------------ R3 (docs/PREREG_R3.md)
R3 = {st: f"go(world_final, grid.r3_preset(BEST, '{st}'), 'r3_{st.lower()}')" for st in ["H1B", "ES", "AGR", "FRAC", "KNOW", "BUDGET2", "ARCH", "BUDGET8"]}
R3_ANALYSIS = r'''
for f in sorted(glob.glob('results/r3_*.shard0.jsonl')):
    st = os.path.basename(f).split('.shard')[0]
    d = A.load(f'results/{st}*.jsonl'); print('\n==', st, len(d), 'runs')
    print(d.pivot_table(index='defense', columns=['attack', 'mal_frac'], values='final_rec_target', aggfunc='mean').round(3).to_string())
'''

# ------------------------------------------------------------------ R4 (docs/PREREG_R4.md)
R4 = {"H1F": "go(world_final, grid.r4_preset(BEST), 'r4_h1f')"}
R4_ANALYSIS = r'''
from scipy.stats import wilcoxon
d = A.load('results/r4_h1f*.jsonl'); print(len(d), 'R4 runs; features:', world_final.n_features)
print(d.pivot_table(index='defense', columns='attack', values='final_rec_target', aggfunc='mean').round(3).to_string())
ps = []
for dn in sorted(d.defense.unique()):
    c = d[(d.defense == dn) & (d.attack == 'none')].set_index('seed').final_rec_target; o = d[(d.defense == dn) & (d.attack == 'opt_adaptive')].set_index('seed').final_rec_target
    i = c.index.intersection(o.index); diff = (c[i] - o[i]).values
    try: p = float(wilcoxon(diff, alternative='greater').pvalue)
    except ValueError: p = 1.0
    ps.append((dn, diff.mean(), p))
o_ = np.argsort([x[2] for x in ps]); m = len(ps); run = 0.0; adj = [0] * m
for k, i in enumerate(o_): run = max(run, (m - k) * ps[i][2]); adj[i] = min(1.0, run)
for (dn, dm, p), q in zip(ps, adj): print(f'H1F {dn:13s} drop {dm:+.3f} p={p:.4f} Holm={q:.4f}')
'''

INTRO = ("Pre-registered in docs/PREREG_REPLICATION.md (committed before launch). The scaler is fitted on tenant training rows only. "
         "NEW seeds for every study. Every stage is guarded: a failure prints and the notebook continues. "
         "*Accelerator = GPU T4 x2, Internet = On; add the Kubernetes dataset (redamorsli/kube-ids0) as an input before a background run.*")


def build(name, title, stages, analysis, run_flags, best=None):
    common = COMMON
    if best is not None:   # pre-registered deviation: use the settings R1 selected instead of repeating the identical tuning
        common = [c for c in COMMON if "go(world_tune" not in "".join(c["source"]) and "Tuning on half A" not in "".join(c["source"])]
        common.append(code("BEST = json.loads('" + json.dumps(best, sort_keys=True) + "')\njson.dump(BEST, open('results/best_params.json', 'w'), indent=1, sort_keys=True)\n"
                           "print('settings selected by R1 (half A, seeds 200-202):', json.dumps(BEST, sort_keys=True))", "R1 settings loaded"))
    cells = [md(f"# {title}\n{INTRO}"), code("RUN = " + repr(run_flags), "switches set")] + common
    for k, body in stages.items():
        cells.append(md(f"## {k}"))
        cells.append(stage(k, body, f"{k} finished (or skipped)"))
    cells.append(md("## Pre-registered analysis (printed here; the paper's tables are regenerated locally by tools/make_tables.py)"))
    cells.append(code(guarded(analysis, "ANALYSIS"), "pre-registered analysis printed"))
    cells.append(code(f"import shutil; shutil.make_archive('results_{name}_bundle', 'zip', 'results')\nprint('files:', sorted(os.listdir('results')))\nprint('FAILED STAGES:', FAILED or 'none')",
                      f"results_{name}_bundle.zip created"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / f"quantum_bc_fl_cids_{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out.name, "with", len([c for c in cells if c["cell_type"] == "code"]), "code cells")


if __name__ == "__main__":
    build("R1", "Replication R1: tuning, main study, ablation, memorizing attacker, confirmatory sealing test", R1, R1_ANALYSIS,
          {k: True for k in R1})
    r1 = ROOT / "results" / "kaggle" / "r1" / "files" / "best_params.json"
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "r3":
        build("R3", "R3: independent attackers, budget, knowledge, malicious share, architecture, 10 more seeds (docs/PREREG_R3.md)",
              R3, R3_ANALYSIS, {k: True for k in R3}, best=json.load(open(r1)))
        raise SystemExit
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "r4":
        old = COMMON
        COMMON = [dict(c, source=[l.replace("batch_rows=100_000, verbose=False)", "batch_rows=100_000, verbose=False, id_filter='fixed')") for l in c["source"]]) for c in COMMON]
        assert any("id_filter='fixed'" in "".join(c["source"]) for c in COMMON), "data cell not patched"
        build("R4", "R4: corrected identifier filter (keeps SRC_TO_DST_* features); primary comparison re-run (docs/PREREG_R4.md)",
              R4, R4_ANALYSIS, {k: True for k in R4}, best=json.load(open(r1)))
        COMMON = old
        raise SystemExit
    build("R2", "Replication R2: optimising attackers, accountability, detection, generalisation, Kubernetes, energy", R2, R2_ANALYSIS,
          {k: True for k in R2}, best=json.load(open(r1)) if r1.exists() else None)
