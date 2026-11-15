# Results map: every headline claim -> experiment, raw records, analysis code, ledger key

Regenerate everything with `python3 tools/make_tables.py` (reads `results/kaggle/r1`, `r2`, `r3`, `r4`), then check the manuscript with
`python3 tools/check_numbers.py`. Ledger = `paper/numbers.json` (key -> value and source file). Pre-registrations: `docs/PREREG_REPLICATION.md`
(git tag `prereg-replication`), `docs/PREREG_R3.md` (tag `prereg-r3`) and `docs/PREREG_R4.md` (tag `prereg-r4`). Independent check of the
H1, gap and R4 tables and of the run counts per study status from the raw records: `python3 tools/audit_tables.py`.

| Claim (manuscript) | Pre-registered as | Raw records | Analysis (`tools/make_tables.py`) | Ledger keys | Table / figure |
|---|---|---|---|---|---|
| Every rule loses DoS recall under the rule-simulating optimiser at 30% (Holm) | H1 (R2), H1b (R3, 20 seeds) | `r2/files/strong*.jsonl`, `r3/files/r3_h1b*.jsonl` | `t_strong`, `t_h1_stats` | `strong.*.drop_opt30`, `strong.*.holm_p`, `h1.*` | T09, T26, fig_optimiser, fig_h1_seeds |
| Main result robust to the corrected identifier filter (37 features) | H1F (R4) | `r4/files/r4_h1f*.jsonl` | `t_r4` (and `tools/audit_tables.py`) | `r4.*` | T32 |
| Larger search budget lowers recall further | H4 (R2), H7 (R3) | `r2/files/strong_x4*`, `r3/files/r3_budget2*`, `r3_budget8*` | `t_strong_extra`, `t_budget` | `strongx.*`, `budget.*` | T24, T28, fig_budget |
| Independent attackers (ES search; published AGR-tailored) | H6 (R3); AGR descriptive | `r3/files/r3_es*`, `r3_agr*` | `t_r3_attacks` | `r3atk.*` | T29 |
| Imperfect knowledge and fewer malicious tenants | H8, H9 (R3) | `r3/files/r3_know*`, `r3_frac*` | `t_r3_attacks` | `r3atk.*` | T29, T30 |
| Other model / local training | ARCH (R3, descriptive) | `r3/files/r3_arch*` | `t_r3_arch` | `arch.*` | T31 |
| Adaptive robustness gap per rule and family | defined in PREREG_R3 | strong, strong_x4, r3 attackers | `t_gap` | `gap.*` | T27, T27b, fig_gap |
| Published min-max with full knowledge is weaker than the optimiser | R2 pre-declared check | `r2/files/strong_minmax_fk*` | `t_strong_extra` | `strongx.*.minmax_full_knowledge` | T24 |
| Sealing beats public probe under memorization | H2 (R1) | `r1/files/settle*.jsonl` | `t_settle` | `settle.*` | T07, T08, fig_settle_seeds |
| Rule-aware attackers: probe rules and CosRep hold, robust aggregators fall | H3 (R1, descriptive) | `r1/files/final*.jsonl` | `t_main` | `main.*` | T03, T04, fig_recall_vs_share |
| Kubernetes: sealed 1.00, public 0.00 under the optimiser | R2 descriptive | `r2/files/kube*.jsonl` | `t_recon_sens_kube` | `kube.*` | T16, fig_kubernetes |
| Slashing violates alpha 2-26x; cause = persistent tenant differences | H5 (R2) | `r2/files/acct*.jsonl` (with per-round traces) | `acct_rules_table`, `t_acct_diagnostics` | `acct.*`, `acctdiag.*` | T10, T23, fig_slashing |
| Implementation of the e-processes is valid under exchangeable scores | - | synthetic | `tools/test_eprocess.py`, `tests/test_core.py` | `results/eprocess_synthetic.json` | text |
| Cross-dataset transfer fails (feature shift, label semantics) | R2 descriptive | `r2/files/loso_*.jsonl`, `loso_diag.json` | `t_loso`, `t_loso_diag` | `loso.*`, `losodiag.*` | T13, T25 |
| Duplicates inflate accuracy (1.8 / 8.5 points) | R2 descriptive | `r2/files/detect_*` | `t_detect` | `detect.dup_inflation.*` | T11, fig_detection |
| ML-DSA-65 adds 29% gas per round | - | `results/chain_gas.json` (`chain/measure_gas.py`) | `t_costs` | `chain.*` | T20, fig_gas |
| Contract properties in the trust table | - | - | `chain/test_contract.py` (11 tests) | - | Table trust |
| Run counts (8,455 federated + 25 references; 2,140 confirmatory, 3,710 pre-specified, 220 checks, 775 exploratory, 1,610 descriptive) | - | all `r1`-`r4` jsonl | `main()` (and `tools/audit_tables.py`) | `runs.*`, `runs.status.*` | Table studies |
| Data cleaning, duplicates, 36/35 features | - | `r1/files/data_report.json` | `t_data` | `data.*` | T01 |

Software: `results/kaggle/r1/files/versions.json`. Checksums: `results/kaggle/SHA256SUMS_replication`.
