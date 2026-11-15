# Pre-registration R3: how far does the adaptive-attack result reach?

Written and committed BEFORE notebook R3 is launched (git tag `prereg-r3`). Same data, split, scaler (training rows only), worlds and
tuned settings as R1/R2 (`results/kaggle/r1/files/best_params.json`). Nothing is re-tuned. Deviations, if any, are appended at the end.

## Motivation
An external review asked (1) whether the main result (H1: a rule-simulating optimiser lowers DoS recall for all 11 rules) is an artefact
of one search procedure, (2) how it changes with budget, attacker knowledge and the malicious share, (3) whether it holds for another
model and training configuration, and (4) whether 10 seeds are enough. R3 answers these; the published optimiser code path is unchanged
(checked: identical results before and after adding the new attackers, `/tmp/cmp.py` run on 2026-10-16).

## New attackers (frozen in this commit, `qbfl/attacks.py`)
- `opt_es` - independent search: (mu, lambda) evolution strategy, 4 generations x 6 candidates = 24 evaluations of the private rule
  copy (same as the optimiser's budget), over SIX directions (poison push, sign flip, mean shift, inverse standard deviation, inverse
  sign, LIE target), coefficients clipped to [0, 8], warm-started from the previous round's best. Same objective as the optimiser.
- `agr_tailored` - the AGR-tailored attack of Shejwalkar and Houmansadr (NDSS 2021), partial knowledge, run against the private rule
  copy: all malicious tenants submit mu + gamma p, p in {inverse unit, inverse std, inverse sign}, gamma in g0 x 2^k, k = -3..4 (24
  evaluations); objective = distance of the aggregate from the benign aggregate (untargeted, as published).
- `opt_noisy` - the optimiser with honest-update stand-ins 4x noisier (noise 2.0 instead of 0.5).
- `opt_lowdata` - the optimiser using one tenth of its own flows (150 instead of 1,500 per class for its objective; 400 instead of
  4,000 for its sealed stand-in probe).
- `opt_adaptive_x2`, `opt_adaptive_x8` - budget variants of the frozen optimiser (x1 and x4 already exist from R2).

## Stages, seeds, runs (1,075 runs)
| Stage | Rules | Attackers | Share | Seeds | Runs |
|---|---|---|---|---|---|
| H1B | 11 | none, opt_adaptive | 30% | 410-419 | 220 |
| ES | 11 | opt_es | 30% | 350-359 | 110 |
| AGR | 11 | agr_tailored | 30% | 350-359 | 110 |
| FRAC | 11 | opt_adaptive | 10%, 20% | 350-359 | 220 |
| KNOW | 11 | opt_noisy, opt_lowdata | 30% | 350-359 | 220 |
| BUDGET2 | 11 | opt_adaptive_x2 | 30% | 350-359 | 110 |
| ARCH | 6 (P2 set) | none, opt_adaptive | 30% | 420-424 | 60 (256-128-64 MLP, 3 local epochs, lr 1e-3) |
| BUDGET8 | median, trust-aware, CosRep, sealed family, public family | opt_adaptive_x8 | 30% | 350-354 | 25 |
Seeds 350-359 pair with the R2 clean runs and R2 optimiser runs of the same seeds (same tenants, malicious set and initial model).

## Hypotheses and tests (one-sided Wilcoxon signed-rank, paired by seed; Holm within each family; 95% bootstrap CIs; effect sizes)
- **H1b (confirmatory):** with 20 seeds (350-359 from R2 + 410-419), recall under the optimiser at 30% is below clean recall for each
  of the 11 rules. Holm over 11. Effect size: mean paired drop with 95% CI and matched-pairs rank-biserial correlation.
- **H6 (confirmatory):** the independent ES attacker lowers recall vs clean for each of the 11 rules (seeds 350-359), Holm over 11.
  Also reported: ES vs optimiser, paired (two-sided, exploratory).
- **H7 (budget, exploratory):** per rule, recall at budgets 1, 2, 4 (and 8 for five rules); Spearman trend over seeds and budgets.
- **H8 (knowledge, confirmatory):** with noisy stand-ins and with one tenth of the data, the optimiser still lowers recall vs clean
  for each rule (Holm over 11 per variant).
- **H9 (share, descriptive + tests):** number of rules with a significant drop (Holm over 11) at 10% and at 20% malicious.
- **AGR-tailored, ARCH (descriptive):** recall, drop vs clean, and the honest false-alarm change.
- **Cost (descriptive):** false-alarm change on benign test flows (final FPR minus clean FPR, paired) and seconds per round for each
  attacker.

## Reporting rule
All results are reported whatever they show. If the independent attackers are clearly weaker than the optimiser, the paper says the
size of the effect depends on the search procedure. If the drop disappears at 10% malicious or for the other architecture, the paper
reports that boundary. The adaptive robustness gap Delta R = R_fixed - R_adaptive is computed per rule, share and budget, where
R_fixed is the worst recall over the attackers that do not simulate the rule (label flip, LIE, min-max, rule-aware, rule-aware with
farming, memorizing) and R_adaptive the worst over the simulating attackers.

## Deviations
(none yet)
