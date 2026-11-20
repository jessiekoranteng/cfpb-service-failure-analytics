# Decommissioning Plan: Retiring the Model Ethically

## 1. Reasons to retire

| Trigger | Notes |
|---|---|
| Panel withdraws approval | Any Gate 3 failure not remedied in one quarter |
| F2 confirmed breach that retraining cannot fix | |
| F3 cannot be met and no baseline alternative exists | Logistic regression is the fallback |
| Source data end or change definition | CFPB file schema change; loss of the tag fields |
| The purpose has gone | Process changed; capacity no longer scarce |
| Cost or risk exceeds benefit | Reviewed yearly |
| Legal change | Act 843 amendment, supervisory instruction |

## 2. Sequence

1. **Decide.** Panel records the reason and the date; informs the service quality lead, DPO, IT.
2. **Stop scoring.** Remove the `/predict` route and disable the stakeholder dashboard's what-if page. Reviewers return to arrival-order reading, which has always been the fallback. A banner on the dashboard states that the model is retired.
3. **Freeze evidence.** Final monitoring run; final model card marked "retired" with the reason; tag the repository release; export the registry (champion, challenger, aliases and tags) and the MLflow run export.
4. **Preserve for accountability.** Keep the Explanation Store, hash verification result and monitoring history for the longer of 12 months and any legal-hold period, so a disputed prioritisation can still be explained. Then purge.
5. **Delete what is no longer needed.** Model bundles in `models/` and `models/archive/`, pseudonymised analysis tables, restricted audit table (protected tags) and the HMAC key. Deletion is logged in the privacy audit chain; the log itself is kept for its retention period (730 days).
6. **Close governance.** Archive the panel log; update the Public Trust Statement ("retired on date, because reason"); end Prometheus scrapes and the scheduled monitoring job.
7. **Learn.** Post-retirement review within 30 days: what the model achieved against goals G1 to G5, what it cost, what complaint handling looks like now.

## 3. What not to do

Do not quietly leave a retired endpoint running; do not keep protected-tag tables "just in case"; do not retrain on the retired model's decisions (reviewer behaviour shaped by scores is not an independent label).

## 4. Responsibilities

Decision: panel. Execution: model owner and IT. Data deletion sign-off: DPO. Communication: service quality lead.
