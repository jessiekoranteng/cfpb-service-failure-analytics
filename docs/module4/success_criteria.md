# Pre-registered success criteria (written before any model was trained)

**Task.** At intake, predict (G1) whether a complaint will close with monetary relief and (G2) whether the company will miss the 15-day response window. Unit of analysis: one complaint. Horizon: the case outcome (closure).

**Split.** Time-aware. Train 2020-01 to 2025-03; validation 2025-04 to 2025-09; test 2025-10 to 2026-06 (touched once). Complaints received after 2026-06-30 are excluded because many are still open (right-censored).

**Primary model (monetary relief), on the test set:** ROC-AUC >= 0.70; PR-AUC >= 2x the test prevalence; recall >= 0.45 when the top 20% of complaints by risk are flagged (review capacity); better than majority-class and logistic-regression baselines.

**Secondary model (untimely response):** PR-AUC >= 3x test prevalence.

**Fairness (protected groups: Older American, Servicemember, each vs everyone else):** disparate impact ratio >= 0.80; equalized-odds difference <= 0.10; demographic-parity difference <= 0.10 at the top-20% flag. If a criterion is missed, apply mitigation and report the accuracy cost.

**Protected attributes available.** Age (via the Older American tag) and military status (Servicemember tag). Gender and race are not in the public data, so they cannot be audited; this is a documented limitation.

---

## Addendum (2026-11-01, written after the test window had been scored; the criteria above are unchanged)

The criteria above were fixed before training. Checking them against `ETHICS_CHARTER.md` (Modules 1-2) afterwards showed that they were **looser** than the charter's fairness objectives, which govern:

* **F2:** true-positive-rate and false-positive-rate gaps <= 0.05 across tag groups (the criteria above allowed 0.10 for the equalized-odds difference).
* **F3:** proxy test: the Older American and Servicemember tags must not be recoverable from the model's features beyond chance (not tested above).
* **F4:** no complaint is de-prioritised below its statutory window (a design property of the service, tested by contract).

These were evaluated on the deployed model (`python -m src.modeling.charter`). Outcome and discussion: `docs/module4/fairness_metrics_report.md`, section 6. The pre-registered results are reported unchanged next to the charter results; nothing was re-tuned in response.

