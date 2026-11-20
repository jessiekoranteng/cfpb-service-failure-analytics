# Limitations Statement

Boundaries and risks of the complaint service-failure risk model. Read with `MODEL_CARD.md`.

## What the model is for, and not for

**For:** suggesting which public-CFPB-style complaints (checking or savings accounts; money transfer, virtual currency or money services; received 2020 to mid-2026) a reviewer might read first.
**Not for:** deciding outcomes, setting deadlines, judging any individual, ranking staff, deciding payouts, or use on any other product, country or period without revalidation.

## Data limits

* Public data only, **U.S. consumers**; the Ghanaian bank framing is a methodology demonstration (risk R6). No local validation exists.
* **The label is a company's decision**, not ground truth. A complaint ending in relief means the company chose to pay, not that harm occurred. Companies differ in policy; the model learns that.
* Text of complaints is not used, so what a complainant actually wrote cannot inform a score. Phone and in-branch complaints reach the model only after a handler enters the nine intake fields; the model does not hear the call.
* Tags (Older American, Servicemember) cover about 15% of complaints; fairness is audited only for them, and only on tagged complaints. Other groups (gender, disability, language) are not auditable with this data.
* Open cases are right-censored: the data window ends four months before the snapshot so cases have time to close. The most recent complaints cannot be scored for outcome yet.

## Model limits

| Limit | Evidence |
|---|---|
| Moderate ranking power | ROC-AUC 0.791 (95% CI 0.786 to 0.795); PR-AUC 0.305, 2.70 times the base rate; about half of relief cases are missed even at the best reading order |
| Skill is company identity | Removing company name and recent volume: ROC-AUC 0.638 (0.594 without sub-issue); SHAP importance of company 64.8% |
| Weak for unseen companies | ROC-AUC 0.626; 18.0% of flags change; the API returns `unseen_values` |
| Sensitive to missing inputs | Missing company volume changes 11.8% of flags, missing sub-issue 8.2% |
| Not better than a simple baseline | Logistic regression PR-AUC higher by 0.011 on test |
| Late-response model unstable | Validation ROC-AUC 0.759, test 0.892; advisory only |
| Calibration depends on the window | ECE 0.0150 on test; the calibrator was fitted on a window with 10.2% relief against 11.3% |
| Threshold boundary effects | Complaints exactly at a regional line (for example 16.3% in the South) are a coin toss; B and D in the worked examples sit there |
| Association, not cause | Reasons are factors that moved a score, not causes of the outcome |

## Fairness limits

* F3 not met: tags can be guessed from the inputs (AUC 0.680 and 0.640).
* F2: Older American TPR gap 0.0467 (interval to 0.075), four-group 0.061; Servicemember passes. Confirmation needs more tagged data.
* Region lines reduce, not remove, the regional gap (recall gap 0.093, limit 0.10) and cost precision; part of the gap reflects real regional base-rate differences (Northeast 13.1% relief against Other 7.6%).
* Mitigation fitted on a three-month calibration window is unstable for small regions: a rehearsal challenger flagged 13% of "Other" complaints against 23% for the Northeast (selection ratio 0.556) and was blocked by the gates.

## Operational limits

Monitoring is a replay. Override behaviour, reading-time effects and real traffic are unobserved. The public API has no authentication. The public demo does not run in Ghana. The ethics panel has not approved the model. No claim about operational savings is made (goal G5 descoped; G3 surge detection built but not back-tested).

## Risks if these limits are ignored

Using scores as decisions; trusting a score for a company the model has not seen; reading a "no flag" as "no problem"; treating a fairness pass at one window as permanent; copying U.S. results into Ghanaian operations without local validation.
