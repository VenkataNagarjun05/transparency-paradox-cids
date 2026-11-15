"""Run a Kaggle notebook end to end on this machine with FAKE data, before spending GPU hours on it.

- pip cells are skipped; every other code cell runs in order in a fresh temp directory (the notebook writes its own qbfl copy);
- D.download is replaced by fake NetFlow-v2-style parquet files (with duplicates, address/port columns, a documentation CSV) and
  fake Kubernetes CSVs;
- every grid is cut to its first two seeds and every run to 2 rounds (structure kept: all rules, all attackers, all stages).
Pass = no 'STAGE FAILED' line, no exception, FAILED == [] and the bundle exists.
Usage: python3 tools/mock_run_notebook.py notebooks/quantum_bc_fl_cids_R1.ipynb
"""
import io
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

FAMS = {"unsw": ["Benign", "DoS", "Exploits", "Backdoor", "Reconnaissance"],
        "cse18": ["Benign", "DDoS attacks-LOIC-HTTP", "Brute Force -Web", "SQL Injection", "Infilteration"],
        "ton": ["benign", "ddos", "scanning", "password", "xss", "backdoor"],
        "bot": ["Benign", "DDoS", "Reconnaissance", "Theft"]}
FEATS = ["PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
         "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
         "RETRANSMITTED_IN_BYTES", "NUM_PKTS_UP_TO_128_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "DNS_QUERY_TYPE",
         "FTP_COMMAND_RET_CODE", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT"]


def fake_sources(root):
    rng = np.random.default_rng(0)
    paths = {}
    for s, names in FAMS.items():
        d = os.path.join(root, s); os.makedirs(d, exist_ok=True)
        rows = []
        for k, n in enumerate(names):
            m = 1400 if k else 2500
            X = np.abs(rng.normal(k * 3 + 1, 2 + k, size=(m, len(FEATS)))) * (10 ** rng.integers(0, 3, len(FEATS)))
            df = pd.DataFrame(X, columns=FEATS)
            df.iloc[: m // 5] = df.iloc[0].values                  # exact duplicates
            df["Attack"] = n; df["Label"] = int(k > 0)
            df["IPV4_SRC_ADDR"] = "10.0.0.1"; df["L4_SRC_PORT"] = 1234; df["IPV4_DST_ADDR"] = "10.0.0.2"; df["L4_DST_PORT"] = 80
            rows.append(df)
        pd.concat(rows).sample(frac=1, random_state=0).to_parquet(os.path.join(d, f"NF-{s}-V2.parquet"))
        pd.DataFrame({"Feature": FEATS, "Description": "x"}).to_csv(os.path.join(d, "NetFlow v2 Features.csv"), index=False)
        paths[s] = d
    k = os.path.join(root, "kube"); os.makedirs(os.path.join(k, "dvwa"), exist_ok=True); os.makedirs(os.path.join(k, "boa"), exist_ok=True)
    kc = [f"f{i}_bytes" for i in range(30)]
    for app, labels, sub in [("dvwa_dataset_ml_ready.csv", [0, 1, 2, 3, 4], "dvwa"), ("boa_dataset_ml_ready_frontend_microservice.csv", [0, 1, 2], "boa")]:
        parts = []
        for lab in labels:
            X = np.abs(rng.normal(lab * 2 + 1, 1.5, size=(900, len(kc))))
            df = pd.DataFrame(X, columns=kc); df["src_ip"] = 1; df["dst_port"] = 30025; df["label"] = lab; parts.append(df)
        pd.concat(parts).to_csv(os.path.join(k, sub, app), index=False)
    paths["kube"] = k
    return paths


def main(nb_path):
    nb = json.load(open(nb_path))
    work = tempfile.mkdtemp(prefix="mockrun_")
    fake = fake_sources(os.path.join(work, "_fake"))
    os.chdir(work); sys.path.insert(0, work)
    ns = {"__name__": "__main__"}
    patched = False
    t0 = time.time(); log = io.StringIO()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("!pip"):
            continue
        if "D.download" in src and not patched:
            D, grid, fl = ns["D"], ns["grid"], sys.modules["qbfl.fl"]
            D.download = lambda key: fake[key]
            orig_run_grid = grid.run_grid

            def small(world, cfgs, out_path, device="cpu", tag=""):
                seeds = sorted({c.seed for c in cfgs})[:2]
                keep = [c for c in cfgs if c.seed in seeds]
                for c in keep:
                    c.rounds, c.eval_every, c.local_epochs = 2, 1, 1
                    c.attack_start = min(c.attack_start, 1); c.farm_rounds = min(c.farm_rounds, 1)
                return orig_run_grid(world, keep, out_path, device, tag)
            grid.run_grid = small
            grid.run_parallel = lambda world, cfgs, out_prefix, devices, tag="": small(world, cfgs, out_prefix + ".shard0.jsonl", "cpu", tag)
            ns["GPUS"] = ["cpu"]
            patched = True
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(src, f"cell{i}", "exec"), ns)
        except Exception:
            print(buf.getvalue()[-3000:]); traceback.print_exc(); print(f"FAILED at cell {i}"); return 1
        out = buf.getvalue(); log.write(out)
        bad = [l for l in out.splitlines() if "STAGE FAILED" in l or "Traceback" in l]
        tail = [l for l in out.splitlines() if "CELL COMPLETED" in l]
        print(f"cell {i:2d} ok {time.time() - t0:6.0f}s {tail[-1].strip() if tail else ''}", *(["\n   " + b for b in bad] if bad else []))
        if "FAILED = []" in src and "GPUS" in ns:
            ns["GPUS"] = ["cpu"]
    failed = ns.get("FAILED", ["?"])
    open(os.path.join(work, "mock_log.txt"), "w").write(log.getvalue())
    bundle = [f for f in os.listdir(work) if f.endswith("_bundle.zip")]
    print("\nFAILED stages:", failed, "| bundle:", bundle, "| log:", os.path.join(work, "mock_log.txt"))
    return 0 if not failed and bundle else 1


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1])))
