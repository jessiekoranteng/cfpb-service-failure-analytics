# Incident Response Plan for Ethical Failures

An ethical failure is any event in which the model, its data or its use may harm complainants, break a promise in the Charter, or break a legal duty. Technical outages are included only where they affect fair handling.

## 1. Severity

| Level | Definition | Examples | Target response |
|---|---|---|---|
| S1 | Probable harm to complainants or breach of law | Scores used to delay or reject complaints; personal data in logs or the dashboard; a confirmed F2 breach persisting after one retrain; F4 breached | Switch the model off within 1 hour of confirmation; panel informed same day |
| S2 | Charter or threshold breach without evidence of harm | Hard alert on accuracy, calibration or regional ratio; F2 point estimate above limit with interval crossing; unseen-company share above 10%; explanation-store verification fails | Owner acts within 1 working day; panel informed in 5 working days |
| S3 | Warning-level signal or noisy alert | Warn alert; single noisy region month | Ticket; recheck at next run |

## 2. Roles

Incident lead: model owner. Decision to switch off: service quality lead (no approval needed). Privacy incidents: DPO leads. Communications: service quality lead with compliance. Panel receives every S1 and S2.

## 3. Steps

1. **Detect.** Prometheus alert, monthly job, a reviewer's report or a complainant's concern.
2. **Triage.** Assign severity; open a ticket; record time, detector and evidence. Retrieve the relevant stored explanations by id and run `GET /explanations/verify`.
3. **Contain.** For S1: stop scoring (stop the API container or remove the route); reviewers return to arrival-order reading; preserve the explanation store and monitoring files unchanged. For S2: freeze threshold changes and retraining promotion.
4. **Investigate.** Decide whether the cause is data (source change, missing tags, a new company), model (drift, calibration) or use (misuse, automation bias). Re-run the monitoring job on the affected window; compare with the previous month.
5. **Remediate.** Options in order: revert to the previous champion (the archive keeps every replaced bundle); revert to one review line; retrain through the gates; withdraw the model.
6. **Notify.** Panel for every S1 and S2; DPO for any personal-data element; affected complainants where the bank's policy or law requires it; the Public Trust Statement is updated if a public promise was broken.
7. **Review.** Blameless post-incident review within 10 working days: timeline, root cause, what detection missed, a new test or alert that would have caught it. Entry in the panel log.

## 4. Scenarios

| # | Scenario | First action | Evidence to gather |
|---|---|---|---|
| 1 | API down | Fall back to arrival-order reading (nothing depends on the API) | Container logs; `ApiDown` alert |
| 2 | Regional selection ratio hard breach for two months | Revert to one line; panel review | Region table, volumes, thresholds |
| 3 | A group is flagged far less than others | Check tag coverage first (was the tag dropped?) then gaps with intervals | Fairness tab; tag coverage by channel |
| 4 | A reviewer or manager uses scores to delay complaints or rank staff | S1: switch off; disciplinary route outside this plan | Access log; stored explanations by user and time |
| 5 | Personal data appear in the explanation store or dashboard | S1: stop, purge affected records, DPO notified | Store schema test; request logs |
| 6 | Explanation store fails verification | S2: treat as tampering until shown otherwise; preserve file | `/explanations/verify` response, first bad index |
| 7 | Sudden score shift after a data change | Check input schema and the Module 3 quality report; rerun monitoring | Evidently report; GX data docs |
| 8 | Public complaint about unfair handling | Retrieve the record by id; explain; log; panel in quarterly review | Stored explanation |

## 5. Tests of this plan

The alert rules have unit tests (`tests/test_monitoring.py`), the store tamper checks have tests (`tests/test_explanation_store.py`), and the promotion guard has tests (`tests/test_retrain.py`). No live incident exercise has been run; a tabletop exercise on scenarios 2 and 4 is required before any pilot.
