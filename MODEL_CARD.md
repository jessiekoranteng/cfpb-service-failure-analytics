# Model Card: CFPB complaint service-failure risk models (v1)

*Format after Mitchell et al. (2019). Numbers are generated from `reports/m4/facts.json`; do not edit by hand.*

## 1. Model details

| Item | Value |
|---|---|
| Owner / author | Jessie Kojo Koranteng (NXU, BAN6800 capstone) |
| Models | **Relief-risk model** (G1: will the complaint close with monetary relief?) and **late-response model** (G2: will the company miss the 15-day window?) |
| Algorithm | LightGBM gradient-boosted trees, isotonic-calibrated; chosen on validation PR-AUC from logistic regression, random forest and LightGBM |
| Version | v1, MLflow registry names `cfpb-monetary_relief-risk` and `cfpb-untimely-risk`, alias `champion` |
| Serialized files | `models/risk_monetary_relief.joblib`, `models/risk_untimely.joblib` (bundle: calibrated model, base pipeline, feature list, thresholds) |
| Inputs (9) | Product, Sub-product, Issue, Sub-issue, Submitted via (channel), region (from state), company, company complaint volume in the previous 4 weeks, month received |
| Outputs | Calibrated probability of monetary relief, calibrated probability of a late response, a review flag, top-3 reasons (SHAP) |
| Data | CFPB open complaint data, pseudonymised through the Module 3 pipeline |

## 2. Intended use

* **Primary use:** decision support for a bank's complaints-quality team (the generic Ghanaian retail-bank framing in Modules 1-2). Rank newly received complaints so that the 20% with the highest relief risk get an early second-line review.
* **Intended users:** complaints analysts and team leads. A person always decides.
* **Status:** research prototype. Not approved for production: the ethics-panel sign-off required by the charter is pending and charter objective F3 is not met (Section 7).
* **Out of scope:** pricing, credit decisions, staff performance ranking, deciding which complainants to delay or deprioritise, any use on individuals' creditworthiness, and any deployment on bank-internal data without re-validation (all data here are U.S. complaints; see Limitations).

## 3. Factors and protected attributes

* **Audited:** Older American tag, Servicemember tag (complainant-reported in the CFPB data) and U.S. region (Northeast, Midwest, South, West, Other).
* **Never model inputs.** Tags are held in a restricted audit table and joined only for evaluation (leakage guard `assert_no_leakage`).
* **Not auditable:** gender, race and ethnicity are not in the public data, so no claim is made about them.

## 4. Data

* **Source:** CFPB Consumer Complaint Database, products "Checking or savings account" and "Money transfer, virtual currency, or money service", received 2020-01-01 onward: 535,601 in-scope records after the Module 3 pipeline.
* **Time-aware split by received week:** train 349,611 (to 2025-03), validation 50,891 (2025-04 to 2025-09), test 94,305 (2025-10 to 2026-06, scored once). Complaints received after 2026-06-30 are excluded because many are still open.
* **Label prevalence:** relief 11.1% train, 10.2% validation, 11.3% test. Late response 1.05% train, 1.99% validation, 1.17% test.
* **Leakage controls:** `Company response to consumer`, `Timely response?` and every other outcome-related column are in a forbidden list; the guard fails the run if one appears. Company volume uses a 4-week lag.

## 5. Performance (test window, calibrated champion)

| Metric | Relief-risk model | Late-response model |
|---|---|---|
| ROC-AUC | 0.791 (95% CI 0.786-0.795) | 0.892 |
| PR-AUC (prevalence) | 0.305 (0.113); lift 2.70x; 95% CI 0.297-0.313 | 0.113 (0.012); lift 9.59x |
| Recall when exactly the top 20% by score are flagged | 0.521 | 0.748 |
| Precision when exactly the top 20% are flagged | 0.295 | 0.044 |
| Brier score | 0.0883 | 0.0111 |
| Expected calibration error | 0.0150 (uncalibrated 0.0069) | 0.0034 (uncalibrated 0.0060) |

At the deployed review threshold (20% capacity rule, threshold 0.175), the relief model flags 18.5% of test complaints with precision 0.306 and recall 0.501.

**Success criteria.** The first twelve rows were pre-registered before training (`docs/module4/success_criteria.md`); the Charter rows (F2, F3 from `ETHICS_CHARTER.md`, Modules 1-2) are stricter than my pre-registered fairness limits and were added afterwards, so they are reported separately and govern:

| Criterion | Target | Observed on test window | Result |
|---|---|---|---|
| Relief ROC-AUC | >= 0.70 | 0.791 | met |
| Relief PR-AUC lift over prevalence | >= 2.0x | 2.700 | met |
| Relief recall when exactly the top 20% are flagged | >= 0.45 | 0.521 | met |
| Relief beats majority-class baseline (PR-AUC) | greater | 0.192 | met |
| Relief beats logistic-regression baseline (PR-AUC, paired bootstrap) | difference > 0 | -0.011 | **NOT MET** |
| Untimely PR-AUC lift over prevalence | >= 3.0x | 9.589 | met |
| Older American: disparate impact ratio | >= 0.80 | 0.869 | met |
| Older American: equalized-odds difference | <= 0.10 | 0.051 | met |
| Older American: demographic-parity difference | <= 0.10 | 0.028 | met |
| Servicemember: disparate impact ratio | >= 0.80 | 0.986 | met |
| Servicemember: equalized-odds difference | <= 0.10 | 0.026 | met |
| Servicemember: demographic-parity difference | <= 0.10 | 0.003 | met |
| Charter F2: TPR and FPR gaps, Older American (deployed) | <= 0.05 | 0.047 | met (95% CI upper bound 0.075; limit not confirmed) |
| Charter F2: TPR and FPR gaps, Servicemember (deployed) | <= 0.05 | 0.017 | met (95% CI upper bound 0.039) |
| Charter F2: TPR and FPR gaps, all four tag groups (incl. 'Both') (deployed) | <= 0.05 | 0.061 | **NOT MET** (95% CI upper bound 0.157; limit not confirmed) |
| Charter F3: Older American tag recoverable from inputs (AUC) | 95% CI includes 0.50 | 0.680 | **NOT MET** (95% CI 0.674-0.686) |
| Charter F3: Servicemember tag recoverable from inputs (AUC) | 95% CI includes 0.50 | 0.640 | **NOT MET** (95% CI 0.633-0.646) |

**Honest note on the missed criterion.** The relief model is not better than the logistic-regression baseline on the test window. The paired-bootstrap difference in PR-AUC is -0.0113 (95% CI -0.0153 to -0.0069) and in recall at capacity -0.0294 (CI -0.0362 to -0.0230). LightGBM was selected because it led on validation PR-AUC (by 0.001, a tie); the test window was touched once and was not used to re-select. The gap is small in practice, and the logistic-regression run is kept in MLflow (run `4b14db46df6f4c3da57293b9a5a30ed3`) as the challenger, and the next scheduled retrain should repeat the selection.

## 6. Explainability

* **Global (SHAP, TreeExplainer):** company identity carries 64.8% of mean absolute SHAP importance, then sub-issue (12.8%), prior-4-week company volume (7.5%), month received (5.8%) and issue (5.5%). Channel and product matter little.
* **Local:** SHAP waterfalls and LIME explanations for three cases (`reports/m4/shap_waterfall_*.png`, `lime_*.png`); the API returns the top-3 reasons per request.
* **Counterfactuals (DiCE):** restricted to plausible changes of sub-issue and channel; one case has no plausible counterfactual (very low-risk company), reported as such.
* **Reading:** the model mainly learns *which company* tends to resolve complaints with relief. That is a historical pattern in company behaviour, not a property of the complainant, and must not be read as a cause.

## 7. Fairness

All results are for the **deployed configuration** (calibrated score compared with a per-region threshold; 19.5% of test complaints flagged) on the test window. Gender and race cannot be audited (not in the data).

| Group | Disparate impact (>= 0.80) | Equalized-odds difference (<= 0.10) | Demographic-parity difference (<= 0.10) |
|---|---|---|---|
| Older American (n = 7,372) | 0.859 | 0.047 | 0.032 |
| Servicemember (n = 8,524) | 0.935 | 0.017 | 0.013 |

These are the limits I pre-registered. The **Ethical AI Charter** (Modules 1-2) is stricter, and its result is mixed:

| Charter F2: gaps <= 0.05 (95% bootstrap CI) | TPR gap | FPR gap | Verdict |
|---|---|---|---|
| Older American | 0.047 (0.016-0.075) | 0.037 (0.027-0.047) | holds on the point estimate only |
| Servicemember | 0.003 (0.001-0.039) | 0.017 (0.008-0.026) | holds, confirmed |
| All four tag groups (incl. Both) | 0.061 (0.034-0.157) | 0.041 (0.031-0.054) | **does not hold** |

| Charter F3: can the tag be recovered from the model's inputs? (chance AUC = 0.50) | AUC from the nine inputs (95% CI) | AUC of the relief score alone | Strongest single input (AUC) |
|---|---|---|---|
| Older American | 0.680 (0.674-0.686) | 0.560 | Company (0.62), Channel (0.55), Issue (0.55) |
| Servicemember | 0.640 (0.633-0.646) | 0.515 | Company (0.59), Region (0.57), Sub-issue (0.52) |

**F3 is not met:** the nine inputs recover both tags well above chance, mainly through the company (complainant populations differ by company), then the channel for Older Americans and the region for Servicemembers. The risk score itself separates the tags only weakly, and error-rate gaps are small, so there is no evidence that the model *acts* on the tags, but the proxy path the charter warns about exists (RAID R-02). **F2** holds on the point estimate for the two single tags, is confirmed for Servicemember, but is not confirmed for Older American (the interval reaches 0.075) and does not hold across the four tag groups, driven by the small "Both" group. With the single global threshold the Older American recall gap was 0.051, just over the limit. **F4** is a design property: the response contains a review flag and reasons only, never a decision or a deadline (contract test `test_api_never_issues_a_decision_or_touches_a_deadline`).

**Region (extra audit):** before mitigation the lowest-to-highest selection-rate ratio was 0.497 and the true-positive-rate gap 0.133 (Northeast flagged most, South least); part of this reflects real differences in relief rates by region. After per-region equal-opportunity thresholds fitted on validation data: ratio 0.875, true-positive-rate gap 0.093, false-positive-rate gap 0.018, flag share 19.5%, precision 0.295, recall 0.509. The thresholds are stored in the bundle and applied by the API. Full detail: `docs/module4/fairness_metrics_report.md`.

## 8. Sensitivity and robustness

* Bootstrap (300 resamples) 95% CI for ROC-AUC 0.786-0.795.
* Company volume +/-20%: ROC-AUC 0.791 / 0.791 (baseline 0.791); flag flips 1.3% / 0.6%.
* Sub-issue not recorded: ROC-AUC 0.765, 8.2% of flags change.
* Company volume missing: ROC-AUC 0.785, 11.8% of flags change.
* **Unseen company:** ROC-AUC falls to 0.626 and PR-AUC to 0.163; 18.0% of flags change. The API reports every unseen category in `unseen_values`.
* Quarterly ROC-AUC on test: 2025Q4 0.802, 2026Q1 0.794, 2026Q2 0.778.
* Removing the January 2025 complaint surge (66,647 rows) from training changes ROC-AUC from 0.791 to 0.792.

## 9. Limitations and risks

1. **Jurisdiction:** all data are U.S. complaints; validity for a Ghanaian bank is untested (RAID R-06).
2. **Label bias:** monetary relief is a *company* decision; the model reproduces company practice, including any inconsistency (R-01).
3. **Company dominance:** about two thirds of importance sits in company identity; new entrants are scored poorly (see unseen-company result).
4. **Drift:** volume and relief rate moved sharply year to year (R-07). Calibration got slightly worse on the test window (ECE 0.0150 vs 0.0069 uncalibrated) because isotonic calibration was fitted on a validation window with a different relief rate.
5. **Late-response model:** the less stable of the two. Its validation-window ROC-AUC (0.759) differs from the test window (0.892), prevalence moved from 1.99% to 1.17%, and the cut-off fixed on validation to flag 20% would flag 40.9% of test complaints because scores shifted with the lower test prevalence (exactly the top 20% by score would catch 75% of late responses). Use its probability as advice only; the API's review flag comes from the relief model.
6. **Misuse (R-04) and automation bias (R-08):** scores must not be used to delay complainants or to rank staff; analysts must see the reasons and can override.
7. **Proxy risk (charter F3, not met):** the company (and, to a lesser degree, channel and region) lets a classifier recover the Older American and Servicemember tags (AUC 0.680 and 0.640). Company is also where most of the model's skill lives, so dropping it is not a fix; the risk is accepted for a prototype only and must be reviewed by the ethics panel before any use.
8. **Group sizes:** the Older American and Servicemember groups are 7-9% of test rows; the intersection ("Both", n = 1,349) is too small for stable estimates.

## 10. Acceptance criteria and monitoring (proposed)

| Signal | Cadence | Alert when |
|---|---|---|
| ROC-AUC, PR-AUC lift (relief) | Quarterly, once labels mature | ROC-AUC < 0.70 or lift < 2.0 |
| Recall at 20% capacity | Quarterly | < 0.45 |
| Expected calibration error | Monthly | > 0.03 |
| Selection-rate ratio by region and tag | Monthly | < 0.80 |
| TPR and FPR gaps by tag, with bootstrap interval | Monthly | point estimate > 0.05 (charter F2) |
| Tag recoverability from inputs (AUC), re-run at each retrain | Each retrain | CI excludes 0.50 (charter F3); escalate to the panel |
| True-positive-rate gap by region and tag | Quarterly | > 0.10 |
| Share of requests with `unseen_values` | Weekly | > 5% |
| Share flagged for review | Weekly | outside 15%-25% |
| Company-mix drift (population stability index) | Monthly | > 0.20 |
| Analyst override rate | Monthly | < 5% (automation bias) or > 60% (model not trusted) |

Retraining: quarterly or when two alerts coincide. Every retrain repeats selection on a fresh validation window, the paired comparison with logistic regression, and the full fairness audit.

## 11. Ethical considerations and caveats

The model supports a human review queue; it does not decide outcomes for individuals. Protected attributes are never inputs, but they are partly recoverable from the inputs that are used (Section 7). Fairness was audited only for attributes present in the data. Region disparity was found and mitigated at a measured cost (Section 7). Charter F3 is not met and F2 is not confirmed for Older Americans, so the model must not be used on live complaints before the ethics panel has reviewed this card. Because a reduction in one gap can widen another, any retrain must re-run the full audit.

## 12. Reproducibility

`python -m src.modeling.train`, `python -m src.modeling.analysis`, `python -m src.modeling.adopt_mitigation`, `python -m src.modeling.compare`, `python -m src.modeling.facts`; seed 42; MLflow experiment `cfpb-service-failure-risk` (51 runs, 37 nested tuning trials).
