# Continuous Fairness Monitoring, Model Drift and Bias Drift Detection

Implemented in `src/monitoring/` (thresholds in `thresholds.py`), shown in `monitoring_app/`, and, for live traffic, in `monitoring/` (Prometheus rules and Grafana). Results below are from the replay of 2025-10 to 2026-06 (94,305 complaints) against a reference window of 2025-04 to 2025-09 (50,891).

## 1. What is measured, how often, and what triggers action

| Signal | Cadence | Warn | Hard | Source of the limit | Replay result |
|---|---|---|---|---|---|
| ROC-AUC | Monthly | < 0.74 | < 0.70 | Module 4 pre-registered | 0.771 to 0.828, never below warn |
| PR-AUC lift | Monthly | < 2.3 | < 2.0 | Module 4 pre-registered | 2.41 to 3.38 |
| Recall at the review line | Monthly | < 0.48 | < 0.45 | Module 4 pre-registered | 0.482 to 0.560 |
| Calibration error (ECE) | Monthly | > 0.03 | > 0.05 | Model Card 0.03; Module 2 KPI 0.05 | 0.005 to 0.026 (whole window 0.0150) |
| Share flagged | Weekly (live), monthly (replay) | outside 15 to 25% | outside 10 to 30% | Model Card | 16.0% to 21.3% |
| Unseen-company share | Weekly | > 5% | > 10% | Model Card | 0.3% to 1.0% |
| Company-mix PSI | Monthly | > 0.10 | > 0.20 | Model Card | **Oct 2025: 0.274 (hard)**; otherwise 0.04 to 0.09 |
| Score PSI | Monthly | > 0.10 | > 0.20 | Same convention | 0.012 to 0.087, never above warn (a first-run artefact is explained in section 4) |
| Regional selection ratio | Monthly | < 0.80 | < 0.70 | Four-fifths rule | 0.69 to 0.90; three months fire on a region with under 300 complaints (noisy) |
| F2 TPR gap, FPR gap by tag | Monthly and quarterly | > 0.05 | > 0.10 | Charter F2; Module 4 regional limit | Older American 0.047 and 0.037, Servicemember 0.003 and 0.017; intervals reported |
| F2 by product, company-size tier, region | Monthly | reported, not graded | | Module 1 asked for these cuts; never produced before | TPR spread across products 0.381, across size tiers 0.615 |
| Analyst override rate | Monthly (live only) | < 5% or > 60% | | Model Card | **Not measurable: replay has no reviewers** |

A confirmed F2 breach is one where the **whole 95% bootstrap interval** (300 resamples) lies above 0.05. A point estimate above 0.05 with an interval that crosses it is reported as "not confirmed" and escalated as a watch item, not as proof.

## 2. Model drift detection

* **Input drift:** Evidently `DataDriftPreset` on Product, Issue, Channel, Region, Company, company volume and the score, reference against replay (20,000 sampled from each). Result: 2 of 7 columns drifted (company volume, distance 0.254 against a 0.10 threshold; company identity, Jensen-Shannon 0.122 against 0.10). Company volume is expected to drift because it counts recent filings.
* **Prediction drift:** the score column (Wasserstein 0.037, not drifted) and score PSI.
* **Performance drift:** monthly ROC-AUC, PR-AUC lift, recall and ECE once cases close. Labels mature about four months after receipt, so performance monitoring lags by that long; drift signals are the early warning.
* **Calibration drift:** the model was calibrated on a window with a 10.2% relief rate and scored a window with 11.3%.

## 3. Bias drift detection

Bias drift means the fairness metrics move even when accuracy does not. The monthly job recomputes, on each month and each quarter:

1. selection-rate ratio across regions (four-fifths screen);
2. TPR and FPR gap, tagged group against everyone else, with bootstrap intervals;
3. tag-group, product, company-size and region tables with cells under 10 positives or negatives suppressed;
4. tag coverage and missingness by channel (a group that stops being tagged disappears from the audit; risk R9).

Quarterly replay results: Older American TPR gap 0.058 (2025Q4), 0.036 (2026Q1), 0.045 (2026Q2); Servicemember 0.051, 0.023, 0.012. The 2025Q4 values exceed 0.05 on the point estimate and are watched, not confirmed.

## 4. What the monitoring found about itself

* The first version of score PSI used reference deciles and raised false alerts in six of nine months (five of them hard), because isotonic calibration produces a step function with many tied scores. It was changed to fixed bins and the artefact disappeared. The change is recorded in `src/monitoring/metrics.py`.
* Regional selection ratio on a single month is noisy for the "Other" region (170 to 240 complaints a month). Alerts raised on a region under 300 complaints are marked noisy and do not count toward the open-alert total or trigger retraining.
* Company identity drifted at the start of the replay window (October 2025), consistent with the company mix recovering after the January 2025 surge (Block Inc was 46% of that month's complaints). No retraining was triggered because no performance signal breached.

## 5. Actions

| Signal level | Action | Owner |
|---|---|---|
| Warn | Ticket; recheck next run | Model owner |
| Hard | Page owner; open incident; freeze threshold changes; evaluate retraining | Model owner, service quality lead |
| Confirmed F2 breach | Escalate to the panel within 5 working days; consider reverting to one line and manual review | Panel |
| Two consecutive warnings on a performance signal, any hard performance breach, company-mix PSI hard in 2 of 3 months, half of columns drifted, or champion older than 6 months | Retraining recommended (`retrain.py decide`, shown on the dashboard) | Model owner |

Current decision (`monitoring_app/assets/retrain_decision.json`): **no retraining triggered**; watch item: company-mix PSI hard in 2025-10.

## 6. Limits

The monitoring is a replay of held-out historical data, not live operation. It cannot see reviewer behaviour, cannot audit groups that are never tagged, and cannot tell whether a gap reflects the model or a real difference in how companies resolve complaints.
