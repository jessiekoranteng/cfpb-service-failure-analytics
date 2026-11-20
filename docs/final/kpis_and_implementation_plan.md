# KPIs and Implementation Plan

## 1. Practical deployment plan

| Phase | Weeks | What happens | Exit criterion | Owner |
|---|---|---|---|---|
| 0. Approvals | 0 to 4 | Panel convened; acceptable-use policy signed; DPO advice on Act 843 s. 41; registration checked; privacy notice drafted | Written panel decision; items 4 to 5, 8 of the sign-off checklist closed | Service quality lead, DPO |
| 1. Local data and validation | 4 to 10 | Bank supplies approved, tokenised complaint records; pipeline run through the Module 3 gates; model refit on local data; F2, F3, calibration re-assessed | Local ROC-AUC and PR-AUC lift meet the pre-registered limits; F3 reassessed; baseline comparison repeated | Model owner |
| 2. Shadow mode | 10 to 18 | Scores computed and stored, **not shown**; reviewers work as now; outcomes matched when cases close | Four months of labels (cases mature slowly); monitoring job on real traffic | Model owner |
| 3. Supervised pilot | 18 to 30 | Scores and reasons shown to one team; override reasons logged; reading order compared with a control team (goal G5, descoped in this project) | Override rate within 5% to 60%; no confirmed F2 breach; reviewers report reasons useful | Team leads |
| 4. Operation | 30 onward | Wider rollout; monthly monitoring; quarterly panel (Gate 3); scheduled retraining check every 6 months | Gate 3 passed each quarter | Panel |

Infrastructure for phases 1 to 4: bank-controlled storage in Ghana (Module 1 specified S3 in af-south-1 and App Runner; delivered work runs on a public demo host, reconciliation log C9), authentication and TLS in front of the API, role-based access to dashboards, the same Compose services for local testing, GitHub Actions for tests and image builds.

## 2. KPIs

### Model and data
| KPI | Target | Frequency | Current (replay) |
|---|---|---|---|
| ROC-AUC | >= 0.70 (warn < 0.74) | Monthly | 0.791 whole window |
| Recall at the review line | >= 0.45 | Monthly | 0.509 at deployed lines |
| Precision of flags | >= base rate x 2 | Monthly | 29.5% against 11.3% base rate |
| Calibration error | <= 0.03 (hard 0.05) | Monthly | 0.0150 |
| Share flagged | 15% to 25% | Weekly | 19.5% |
| Unseen-company share | <= 5% | Weekly | 0.3% to 1.0% |
| Pipeline quality checks passed | 37 of 37 | Each run | 37 of 37 (Module 3) |
| API p95 scoring time | <= 0.5 s | Continuous | Not measured on a live host |

### Fairness and governance
| KPI | Target | Frequency | Current |
|---|---|---|---|
| Regional selection ratio | >= 0.80 | Monthly | 0.875 (window); monthly 0.69 to 0.90 |
| F2 gaps by tag | <= 0.05, interval reported | Monthly and quarterly | Older American 0.047 (not confirmed), Servicemember 0.017 |
| F3 tag recoverability | AUC interval includes 0.50 | Each retrain | 0.680 and 0.640 (not met) |
| Alerts open more than 30 days | 0 | Monthly | 0 |
| Explanation-store verification | ok | Weekly | ok in tests |
| Override rate | 5% to 60% | Monthly | Not measurable yet |
| Panel gates passed on schedule | 100% | Quarterly | 0 (no panel) |

### Business (to be measured in the pilot, none claimed now)
Share of relief cases read within one working day of receipt; time to first read of flagged complaints; complaint-handling satisfaction. The project did **not** measure operational savings (goal G5 descoped) and makes no such claim.

## 3. Governance in operation

Monitoring review monthly (model owner to service quality lead); panel quarterly; incident plan on trigger; retraining only through the gates with a named approver and ticket; every change recorded in `CHANGELOG.md`.

## 4. Dependencies and risks to delivery

Local data access (DPO and legal clearance); local labels (Ghana complaint outcomes may not mirror U.S. relief outcomes, risk R6); panel availability; reviewer time for the pilot; maturity lag of labels (about four months). Costs have not been estimated.
