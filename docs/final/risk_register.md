# Risk Register (Module 1 R1 to R8, updated; R9 to R11 added)

L = likelihood, I = impact (Low, Medium, High). "Status" is where the risk stands at the end of the Final Project.

| ID | Risk | L / I | Mitigation promised | What was actually done | Status |
|---|---|---|---|---|---|
| R1 | Label bias: relief decisions vary by company and may under-compensate some groups; the model learns the inequity | M / H | Audit TPR and FPR by tag and company; **train with and without Company**; Fairlearn mitigation; state label caveat | Tag audit and region mitigation done in Module 4. The with/without-company run was **not done** until the Final Project: without company, ROC-AUC 0.638 (from 0.791). Label caveat is in the Model Card | **Open**: not removable with this data |
| R2 | Proxy discrimination through state, company or issue encoding protected status | M / H | Drop ZIP; aggregate state to region; run F3 | ZIP dropped, region used, F3 run: **not met** (AUC 0.680 and 0.640); company is the carrier | **Open, unmitigated** |
| R3 | Re-identification from small cells | L / H | No ZIP or narrative; suppress cells < 10 | k >= 5 on the table; cells < 10 suppressed in monitoring; explanation store holds only nine fields | Controlled |
| R4 | Misuse: scores used to delay complainants or rank staff | M / H | Scores reorder review only (F4); acceptable-use policy; access logging | F4 contract test; API returns no decision; explanation store records every score; **policy not signed; no access log on the public demo** | Partly controlled |
| R5 | Regulatory non-compliance when bank data is introduced (Act 843) | M / H | DPO approval and tokenisation first | Compliance report and open actions list; no real data used | Controlled for now |
| R6 | Overclaiming U.S.-to-Ghana validity | H / M | Methodology demonstration; validate only on approved anonymised bank records; state limit in every output | Stated in dashboard, model card, limitations statement; **no local validation** | **Open** |
| R7 | Drift: yearly volume rose 2.4x from 2024 to 2025 | H / M | Rolling out-of-time validation; drift alerts; retraining trigger | Monthly Evidently and PSI job, alert rules, `retrain.py decide` and gated challenger | Controlled (replay-tested) |
| R8 | Automation bias: staff accept scores uncritically | M / M | Reason codes shown; override reasons logged and reviewed | Reasons shown; override signal defined; **no reviewers, so not measured** | **Open** |
| R9 (new) | Fairness audit covers only tagged complaints; tag coverage is 15.1% and varies from 3% to 57% by channel | H / M | n/a (surfaced by Module 3 bias suite) | Coverage and missingness monitored by channel; confidence intervals reported; groups outside the tags are unauditable | **Open** |
| R10 (new) | Skill concentrated in company identity (SHAP 64.8%); five companies hold 45.8% of complaints; unseen companies fall to ROC-AUC 0.626 | H / M | n/a (surfaced by Modules 3 and 4) | `unseen_values` on every score; unseen-share alert; company-mix PSI; routing of unseen companies to manual review | Partly controlled |
| R11 (new) | Public API unauthenticated and unthrottled; abuse or scraping of the scoring endpoint | M / L | n/a | Requests hold no personal data; no secrets; 90-day store; authentication required before any non-public hosting | Accepted for the demo |

Module 2's RAID log (13 risks) holds R-01 (label bias, 20), R-07 (drift, 20), R-04 (misuse, 15) and R-10 (target leakage, 15); R-10 is closed by the feature allow-list test.
