# Pre-registration: replication after the scaler-leakage fix

Written and committed to git BEFORE the replication notebooks (R1, R2) are launched. The commit hash of this file is recorded in
the paper. Nothing below is changed after results are seen; any later deviation is listed in a "Deviations" section appended at
the end with its date and reason.

## Why a replication
An external review (2026-10-16) found that the feature scaler was fitted on pooled flows that included the test rows (no labels used,
but test-distribution information leaked into preprocessing). The scaler is now fitted on the tenants' training rows only and frozen
for the probe pool and test sets (`qbfl/data.py`, `scaler_fit="train"`). Because this touches every result, every study is re-run.
The replication also answers the review's second point: the strongest attacker was designed after the main study, so its result is
now confirmed on new seeds with its specification frozen here.

## What is fixed in advance
- **Code:** the git commit that contains this file. Notebooks: `notebooks/quantum_bc_fl_cids_R1.ipynb`, `..._R2.ipynb`, generated
  by `tools/build_replication.py` from that commit, and mock-run end to end on fake data (`tools/mock_run_notebook.py`) before launch.
- **Data:** same sources, cleaning, deduplication and half A / half B split (seed 12345); worlds with seed 1 (tuning) and 0 (final).
- **Tuning:** same grids (`qbfl/grid.py` TUNE_GRIDS), same attacks, same selection rule (`analysis.select_best`), on half A only,
  seeds 200-202. R1 and R2 each re-derive the settings; they must agree (checked; a disagreement is reported as a deviation).
- **Seeds (all new):** main 300-309, memorize 310-319, settle 320-349, strong 350-359, leave-one-source-out 360-364,
  reconnaissance 365-369, accountability 370-379, sensitivity 380-384, detection 390-394, Kubernetes 400-404, energy 500.
- **Optimising attacker (frozen):** `qbfl/attacks.py::_opt_adaptive` as in this commit: search family
  u = h + a1 (p - h) - a2 h - a3 |h| mean(h)/|mean(h)|; 11 fixed shapes + previous best + 12 random draws (log-uniform [0.05, 8], each
  component off with probability 0.4); objective = drop in own-sample target recall minus 0.5 x (false-alarm rise above 2 points);
  private rule copy with the public state copied each round; for sealed rules a stand-in probe of up to 4,000 own flows.
  Variant `opt_adaptive_x4` (4x random draws + 36 local refinements) is a pre-declared sensitivity check, not a new primary test.
- **Published attacker added:** min-max (Shejwalkar and Houmansadr, NDSS 2021) with full knowledge of the honest updates
  (`see_benign=True`), i.e. the attacker that commit-reveal is meant to stop. Exploratory.

## Hypotheses and tests
Primary metric everywhere: target-family (DoS) recall of the final global model on half-B test sets. Paired by seed.
One-sided Wilcoxon signed-rank; Holm correction within each family of tests; 95% percentile bootstrap CIs (10,000 resamples).

- **H1 (main claim, transparency paradox).** For each of the 11 rules in follow-up 2B, recall under `opt_adaptive` at 30% malicious
  is lower than clean recall (seeds 350-359). Holm over the 11 rules, alpha = 0.05.
  - Confirmed if all 11 are significant. If k < 11 are significant, the paper's claim is narrowed to those k rules and says so.
- **H2 (sealing).** Under `probe_memorize` at 30% (seeds 320-349): primary sealed_fam - public_fam > 0; secondary sealed_fam -
  probe_public > 0 and sealed_abs - probe_public > 0. Holm over the three. (This is the same primary test as before; the earlier
  run did not confirm it, adjusted p = 0.124.)
- **H3 (rule-aware attacker, descriptive).** Recall at 10/20/30/40% malicious under the rule-aware attackers, per rule (seeds 300-309),
  with the paired tests of the sealed family rule against median, trimmed mean, public probe, public family and sealed abs.
- **H4 (attack budget, sensitivity).** Recall under `opt_adaptive_x4` vs `opt_adaptive` at 30% (seeds 350-359). Reported per rule;
  if the larger budget lowers recall further, the paper reports the larger-budget numbers as the headline.
- **H5 (accountability).** Honest tenants ever slashed for each e-process rule vs its alpha (seeds 370-379, 95% Clopper-Pearson upper
  bound). Diagnostics from the per-round p-value traces: (a) per-tenant mean peer p-value (persistent heterogeneity),
  (b) Kolmogorov-Smirnov distance of pooled honest p-values from uniform by p-value type, (c) lag-1 autocorrelation and trend of each
  honest tenant's self/raw p-values (non-stationarity). The implementation itself is checked on synthetic data
  (`tools/test_eprocess.py`, all rules within alpha under H0).
- **Robustness (descriptive):** reconnaissance target, alpha in {0.3, 10}, 50 tenants, leave-one-source-out with diagnostics
  (family mix, unseen families, feature shift, per-family held-out recall), Kubernetes, detection quality, energy, MIA.

## What the paper will report
The replication results replace the earlier results everywhere in the paper (the earlier pipeline had the scaler leak). The earlier
results are summarised in one appendix paragraph only to show whether the conclusions changed. Every table and figure is regenerated by
`tools/make_tables.py` from the replication archive; `tools/check_numbers.py` must report 0 untraced numbers.

## Deviations
(none yet)
- 2026-10-15 (before any replication result was seen): Kaggle GPU quota left this week was 14h06m, too little to run R1 and R2 in
  parallel with duplicated tuning (~11.5 h). R1 runs first; R2 is then rebuilt with the settings R1 selected (results/best_params.json
  from R1) instead of re-deriving them. Effect: one tuning run instead of two; no change to any hypothesis, seed or test.
