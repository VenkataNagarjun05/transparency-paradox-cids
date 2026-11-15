# Pre-registration R4: does the identifier-filter error change the primary result?

Written and committed BEFORE notebook R4 is launched (git tag `prereg-r4`).

## Why
The column filter used in R1-R3 drops identifier columns with a prefix rule that also matched two traffic features,
SRC_TO_DST_SECOND_BYTES and SRC_TO_DST_AVG_THROUGHPUT, while their DST_TO_SRC_* counterparts were kept (disclosed in the manuscript).
A reviewer asked for the experiments to be re-run with the feature list corrected. R4 re-runs the primary comparison with the corrected
filter (`qbfl/data.py`, `id_filter="fixed"`, pattern `^SRC_IP` instead of `^SRC_`); the original filter stays the default so R1-R3 reproduce.

## What is fixed
- Data: same sources, cleaning, deduplication, split (seed 12345), worlds (seed 0 final), scaler on tenant training rows only; the only
  change is the filter, which adds the two SRC_TO_DST_* features (expected: 37 shared features instead of 35).
- Settings: those selected by R1 on half A (`results/kaggle/r1/files/best_params.json`), not re-tuned, so that R4 isolates the effect of
  the feature correction.
- Runs: the 11 rules of H1 x {clean, rule-aware, memorizing, optimiser} at 30% malicious, seeds 350-359 (440 runs), the same seeds as
  the R2 study it repeats. The optimiser code path is the frozen one (`opt_adaptive`).

## Tests
- **H1F (confirmatory):** for each of the 11 rules, recall under the optimiser is below clean recall (one-sided Wilcoxon over the 10
  seeds, Holm over 11 rules), on the corrected features.
- **Comparison with R2 (descriptive):** per rule, clean and optimiser recall and the adaptive gap (worst of rule-aware and memorizing
  minus optimiser) with the corrected and the original features, with the mean difference and its bootstrap CI. Runs of the two pipelines
  share seeds but not features, so differences are reported, not tested as paired effects of one treatment.

## Reporting rule
If H1F holds for all 11 rules, the paper states that the main result is robust to the filter error. If it fails for some rules, the paper
reports those rules and narrows the claim; the corrected pipeline's numbers are then shown next to the original ones.

## Deviations
(none yet)
