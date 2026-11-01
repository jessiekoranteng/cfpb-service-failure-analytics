# Fairness Metrics Report: relief-risk model (v1)

*Generated from `reports/m4/facts.json`. Library: Fairlearn `MetricFrame`, `demographic_parity_difference/ratio`, `equalized_odds_difference`; disparate impact ratio = lower selection rate divided by higher selection rate, so it is below 1 whichever group is flagged more often. Test window 2025-10 to 2026-06, flags at the deployed threshold.*

## 1. Fairness objectives (from Modules 1-2)

The Module 1 Vision and Module 2 Project Overview commit the project to **fair prioritisation (G4)**: the review queue must not systematically under- or over-select complainants because of who they are. Fixed before training (`docs/module4/success_criteria.md`): disparate impact ratio >= 0.80, equalized-odds difference <= 0.10, demographic-parity difference <= 0.10, for each protected group against everyone else. If a criterion is missed: mitigate and report the accuracy cost.

**Protected attributes available:** the Older American tag (age 62+) and the Servicemember tag. Gender, race and ethnicity are absent from the public data and **cannot be audited**; no fairness claim is made about them. Region (U.S. Census region from the state code) was audited as an additional, non-pre-registered dimension because Module 3's bias suite could block on it.

## 2. Data bias carried in from Module 3

The Module 3 bias suite ran on 535,601 records with overall status **WARN**:

* **Coverage:** tags are recorded on only 15.1% of complaints (the rest "None recorded"), so a tag audit has a recording-bias caveat. Fax (n = 129) and Email (n = 7) are too small to audit.
* **Representation drift** across years: largest shifts in channel mix (Web 22.5% change in share) and product mix (16.0%), and a smaller shift for South (6.7%).
* **Differential missingness:** the share of missing fields varies by channel from 3.1% to 57.1%, but not meaningfully by region (gap 3.8%).
* **Outcome disparity in the raw labels** (before any model): relief rates by region run from 7.6% (Other) to 13.1% (Northeast), a ratio of 0.582; by tag group the ratio is 0.731. These are real differences in the historical outcomes, so a model that predicts them well will select some groups more often than others. This is why region selection rates in Section 4 differ *before* any model bias is considered, and why equal-opportunity (equal recall among true positives) was preferred to equal selection rates.
* **Concentration:** the top five companies account for 45.8% of records (1,364 companies, HHI 0.055). Relief rates span 0%-33% among companies with 1,000+ complaints (RAID R-01), which is why company identity dominates the model.

## 3. Results for the pre-registered groups

| Stage | Group | n in group | Selection rate (group / rest) | TPR (group / rest) | FPR (group / rest) | Disparate impact | Equalized-odds diff | Demographic-parity diff |
|---|---|---|---|---|---|---|---|---|
| Before mitigation | Older American | 7,372 | 21.0% / 18.3% | 0.454 / 0.505 | 0.175 / 0.142 | 0.869 | 0.051 | 0.028 |
| Before mitigation | Servicemember | 8,524 | 18.7% / 18.5% | 0.477 / 0.503 | 0.152 / 0.144 | 0.986 | 0.026 | 0.003 |
| After region thresholds | Older American | 7,372 | 22.5% / 19.3% | 0.466 / 0.513 | 0.189 / 0.153 | 0.859 | 0.047 | 0.032 |
| After region thresholds | Servicemember | 8,524 | 20.8% / 19.4% | 0.506 / 0.509 | 0.171 / 0.154 | 0.935 | 0.017 | 0.013 |

**Reading.** All pre-registered criteria hold before mitigation. Older Americans are flagged slightly *more* often (21.0% vs 18.3%) but caught slightly *less* often when relief was due (TPR 0.454 vs 0.505) and more often wrongly (FPR 0.175 vs 0.142): a small recall gap of 0.051, inside the limit but the one to watch. The late-response model gives disparate impact ratios of 0.983 (Older American) and 0.902 (Servicemember); the Servicemember value is closest to the 0.80 line and is a monitoring item.

Combined tag groups (before mitigation):

| Tag group | n | Selection rate | TPR | FPR |
|---|---|---|---|---|
| Both | 1,349 | 20.1% | 0.457 | 0.160 |
| None recorded | 79,758 | 18.3% | 0.507 | 0.141 |
| Older American | 6,023 | 21.2% | 0.454 | 0.178 |
| Servicemember | 7,175 | 18.5% | 0.482 | 0.150 |

Intersection "Both" is small (n = 1,349); estimates are unstable and are not used for decisions.

## 4. Region disparity: detected

| Region | n | Selection rate | TPR | FPR |
|---|---|---|---|---|
| Midwest | 14,097 | 16.9% | 0.483 | 0.135 |
| Northeast | 16,689 | 24.1% | 0.600 | 0.187 |
| Other | 2,039 | 12.0% | 0.486 | 0.085 |
| South | 39,478 | 16.8% | 0.467 | 0.133 |
| West | 22,002 | 19.0% | 0.483 | 0.146 |

Selection-rate ratio (lowest/highest) = **0.497** and TPR gap = **0.133**; both outside the limits used for the tags. The Northeast is flagged at 24.1% with recall 0.600, against the South at 16.8% and 0.467. Region is not a model input as a protected attribute, but the model receives it, and company mix differs by region (a proxy path, RAID R-02).

## 5. Mitigation options compared

| Option | TPR gap | FPR gap | Selection ratio | Share flagged | Precision | Recall |
|---|---|---|---|---|---|---|
| None | 0.133 | 0.102 | 0.497 | 18.5% | 0.306 | 0.501 |
| Fairlearn ThresholdOptimizer (equalized odds across region) | 0.095 | 0.040 | 0.920 | 34.9% | 0.233 | 0.719 |
| **Per-region equal-opportunity thresholds (adopted)** | 0.093 | 0.018 | 0.875 | 19.5% | 0.295 | 0.509 |

ThresholdOptimizer closes the selection-rate gap but raises the flagged share from 18.5% to 34.9%, which nearly doubles the review workload against a 20% capacity: **rejected as operationally infeasible**. The adopted method keeps the workload (19.5%) and equalises recall across regions by choosing one threshold per region on the *validation* window (equal-opportunity target recall 0.567); thresholds are applied on the untouched test window.

Per-region equal-opportunity results:

| Region | n | Selection rate | TPR | FPR | Threshold |
|---|---|---|---|---|---|
| Midwest | 14,097 | 19.9% | 0.517 | 0.164 | 0.163 |
| Northeast | 16,689 | 21.0% | 0.542 | 0.160 | 0.210 |
| Other | 2,039 | 18.3% | 0.576 | 0.146 | 0.125 |
| South | 39,478 | 19.1% | 0.503 | 0.156 | 0.163 |
| West | 22,002 | 19.0% | 0.483 | 0.147 | 0.165 |

**Cost and residual risk.** Overall precision changes from 0.306 to 0.295 and recall from 0.501 to 0.509. The TPR gap falls from 0.133 to 0.093 (inside 0.10, but only just) and the selection ratio rises to 0.875. Tag results stay inside every limit after mitigation (Section 3). Residual risks: (1) "Other" has n = 2,039, so its threshold (0.125) is noisy; (2) West still has the lowest recall (0.483); (3) per-group thresholds are a policy choice that treats groups differently on purpose, which must be approved and documented by governance before use.

## 6. Objectives of the Ethical AI Charter (Modules 1-2)

The charter (`ETHICS_CHARTER.md`) sets stricter fairness objectives than the limits I pre-registered. They govern, and I evaluated them after the test window had been scored (see the addendum in `docs/module4/success_criteria.md`); nothing was re-tuned in response.

**F2: true-positive-rate and false-positive-rate gaps <= 0.05 across tag groups** (gap, 95% bootstrap interval from 300 resamples of the test window):

| Single global threshold | TPR gap | FPR gap | Verdict |
|---|---|---|---|
| Older American | 0.051 (0.017-0.081) | 0.032 (0.023-0.042) | **does not hold** |
| Servicemember | 0.026 (0.002-0.057) | 0.008 (0.000-0.016) | holds on the point estimate only |
| All four tag groups (incl. Both) | 0.054 (0.037-0.125) | 0.036 (0.026-0.048) | **does not hold** |

| Deployed per-region thresholds | TPR gap | FPR gap | Verdict |
|---|---|---|---|
| Older American | 0.047 (0.016-0.075) | 0.037 (0.027-0.047) | holds on the point estimate only |
| Servicemember | 0.003 (0.001-0.039) | 0.017 (0.008-0.026) | holds, confirmed |
| All four tag groups (incl. Both) | 0.061 (0.034-0.157) | 0.041 (0.031-0.054) | **does not hold** |

*Reading.* The Servicemember gaps are small and the limit is confirmed. For Older Americans the global threshold missed the limit by 0.0009 and the deployed thresholds meet it on the point estimate, but the interval is wide (the group has 7,372 test rows), so **the 0.05 limit cannot be confirmed**. The four-group view fails (0.061) because the "Both" group (n = 1,349) has a recall of 0.457, against 0.512 for complaints with no tag. An honest reading is that gaps of this size are not distinguishable from noise with the available data, not that the model is fair at 0.05.

**F3: proxy test.** A LightGBM classifier trained on the model's own nine inputs to predict each tag (150,000 training complaints, scored on the test window):

| Tag | AUC from the nine inputs (95% CI) | AUC of the relief score alone | Strongest single input (AUC) |
|---|---|---|---|
| Older American | 0.680 (0.674-0.686) | 0.560 | Company (0.62), Channel (0.55), Issue (0.55) |
| Servicemember | 0.640 (0.633-0.646) | 0.515 | Company (0.59), Region (0.57), Sub-issue (0.52) |

*Reading.* **F3 is not met.** Both tags are recoverable well above chance (0.50). The risk score alone carries much less tag information, which is why the error-rate audit above is mostly clean, but the inputs are proxies (RAID R-02) and the model could come to use them if the company mix changes. Removing the main proxy is not a fix: company identity carries about two thirds of the model's SHAP importance, so dropping it would remove most of the predictive skill.

**F4: no complaint is de-prioritised below its statutory window.** Design property. The API returns a review flag, probabilities and reasons only; there is no decision, deadline or ranking-below-window output, and a contract test fails if one is added.

**Recommendation.** (1) Treat the model as a research prototype; (2) ask the ethics panel to rule on whether F3 should be read as written or restated as a bound on how much the *score* depends on the tags; (3) monitor F2 monthly with its interval and escalate when the point estimate exceeds 0.05; (4) do not add tag-specific thresholds, which would use the tags as decision inputs and breach charter F1.

## 7. What this audit does not show

Tags are self-reported on a minority of complaints; absence of a tag is not absence of the characteristic. Equalised error rates do not make the *labels* fair: relief is a company decision, so the model can be perfectly consistent and still copy inconsistent company practice. No gender or race audit is possible. Fairness metrics are computed on a single test window; confidence intervals for the smaller groups are wide.
