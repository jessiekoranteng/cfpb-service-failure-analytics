# Ethical AI Framework

Complaint service-failure risk model (BAN6800 capstone, Modules 1 to 5 and Final Project)
Status: **research prototype. Not approved for operational use.** Ethics-panel sign-off is pending (see `ethical_governance_structure.md`).

## 1. Scope and purpose

The system ranks closed-case-type complaints so that a trained reviewer reads the ones most likely to end in monetary relief first. It issues a
score, a flag and three plain-language reasons. It never decides redress, never changes a statutory response window, and never acts on an
individual. The data are the public CFPB Consumer Complaint Database; the bank setting is a generic Ghanaian retail bank, used to frame the method.
No confidential bank data were used.

## 2. Principles and where each one is enforced

| Principle (Charter) | Commitment | Enforced by | Evidence | Status |
|---|---|---|---|---|
| Fairness | Equalized odds is the primary criterion; demographic parity is a diagnostic | Fairlearn audit at every release; monthly replay job; challenger fairness gates | `reports/m4/charter_checks.json`, `monitoring_app` Fairness tab, `src/modeling/retrain.py` | Partly met (section 4) |
| Transparency | Every score ships with reasons; model card and datasheet versioned with the model | `Scorer.score` reason codes; Explanation Store; `MODEL_CARD.md`; dashboards | `tests/test_explanation_store.py`, `tests/test_modeling.py` | Met for reasons and record keeping; model not self-explaining (company dominates) |
| Accountability | A named owner and ethics-panel sign-off for each model version; the model reorders attention only | Promotion needs approver and ticket; sign-off checklist | `retrain.promote`, `tests/test_retrain.py` | Mechanism built; **panel approval pending** |
| Proportionality | Only what the task needs: 9 inputs, no names, ZIP, narrative or tags | Feature allow-list and leakage test; k-anonymity k >= 5 in the pipeline | `tests/test_modeling.py`, 46 pipeline tests | Met |

Fairness objectives F1 to F4 (Charter): F1 tags are audit-only (met by construction and test); F2 TPR and FPR gaps <= 0.05 (partly met, see below);
F3 tags not recoverable from inputs (**not met**); F4 no complaint is de-prioritised below its statutory window (met by design: the API returns no
deadline or decision, and the contract test asserts that).

## 3. Lifecycle controls

| Stage | Control | Owner |
|---|---|---|
| Data | Public source only; 12 of 15 columns; hash-chained privacy audit log; Great Expectations (37 checks) can block publication; protected tags held in a restricted audit table | Data owner, DPO |
| Build | Pre-registered success criteria written before training; time-aware split with the test window scored once; leakage guard | Model owner |
| Validate | 17 pre-registered checks (13 met); paired bootstrap against a logistic baseline; sensitivity tests; ablation without company identity | Model owner, independent reviewer |
| Release | Three ethics gates (data, release, quarterly); model card; fairness report; twelve retraining gates for any new version | Panel |
| Operate | Explanation Store on every request; Prometheus alerts; monthly Evidently and Fairlearn job; human decides every case | Service quality lead |
| Retire | Decommissioning plan (`decommissioning_plan.md`) | Panel, DPO |

## 4. Where the ethics work fell short (reported, not softened)

1. **F3 is not met.** A classifier can recover the Older American tag from the model's own nine inputs with AUC 0.680 (95% CI 0.674 to 0.686) and the Servicemember tag with 0.640. Company identity is the main carrier. The model was documented as such rather than adjusted away.
2. **F2 is only partly met.** At the deployed region lines the Older American TPR gap is 0.0467 and the FPR gap 0.0366: under 0.05 on the point estimate, but the TPR interval reaches 0.075, so compliance is not confirmed. The four-group comparison (including the 1,349 complaints tagged both) is 0.061 and fails. Servicemember passes (0.017, interval to 0.039).
3. **Mitigation traded precision for balance.** One review line for everyone gave a regional selection ratio of 0.497 (Northeast flagged far more than Other). Per-region lines raised it to 0.875 and cut the recall gap from 0.133 to 0.093, at the cost of precision (about 29.5% at the deployed lines). Fairlearn's ThresholdOptimizer was tried first and rejected: it equalised error rates by flagging 34.9% of complaints against a 20% review capacity.
4. **The model's skill is company identity.** Retrained without company name and recent company volume, test ROC-AUC falls from 0.791 to 0.638; also without sub-issue, 0.594 (`reports/final/ablation_company.json`). Part of the apparent fairness work is therefore about how companies resolve complaints, which is a label problem (a company's decision, not ground truth), not only a model problem. An unseen company scores ROC-AUC 0.626.
5. **A simple baseline is as good.** Logistic regression beats the chosen LightGBM on PR-AUC by 0.011 (paired bootstrap interval excludes zero). LightGBM was chosen on validation before the test window was opened and is reported honestly.
6. **Tag coverage is about 15%.** Fairness is audited on tagged complaints only; missingness varies from 3% to 57% by channel. A group that is never tagged is never audited (risk R9).
7. **Replay, not live.** Monitoring numbers come from replaying a held-out window; no human review or override has been observed, so automation bias (risk R8) is unmeasured.
8. **No ethics-panel approval exists.** The panel's terms and checklist are defined; no approval is recorded and none is claimed.

## 5. Human oversight

A reviewer sees the score, the flag, the review line for the complaint's region and three reasons, and always decides. Reviewers can override and must give a reason; override rate is a monitored signal (alert below 5%, which suggests automation bias, or above 60%, which suggests the model is not trusted). Complainants are not told a score exists; see `public_trust_statement.md` for what the institution commits to say publicly. A person affected by a decision can ask for the stored explanation by its id.

## 6. Explanation, record keeping and redress

* **Global:** SHAP importance in business terms (company 64.8%, sub-issue 12.8%, company volume 7.5%, month 5.8%, issue 5.5%).
* **Local:** top three SHAP reasons on every score; LIME cross-check on three worked cases (the promised 200-case sample was not run, reconciliation log C16); counterfactuals (DiCE) for four cases.
* **Explanation Store:** every `/predict` is written to a hash-chained log with inputs, model version, scores, flag, review line and reasons; `GET /explanations/{id}` retrieves it and `GET /explanations/verify` detects edits or deletions; 90-day retention (so older explanations cannot be reproduced from the log; the model version and inputs allow re-scoring).
* **Redress route:** a complainant or reviewer who disputes a prioritisation raises it with the service quality lead, who retrieves the record by id and reports to the panel quarterly. No automated decision is ever the only basis for an outcome.

## 7. Review cycle

Framework reviewed every 6 months and after any hard alert, incident, retraining or change of data source. Changes are recorded in `CHANGELOG.md` and the panel log in `ETHICS_CHARTER.md`.
