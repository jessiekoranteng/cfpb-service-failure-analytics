# Ethical AI Charter

Carried forward from the Module 1 Vision Document and extended in Module 2.

1. **Fairness.** Equalized odds is the primary criterion; demographic parity is reported as a diagnostic because base rates differ by group.
2. **Transparency.** Every score ships with SHAP reason codes; datasheet and model card are versioned with the model.
3. **Accountability.** Each model version has a named owner and ethics-panel sign-off. The model reorders human attention; it never changes a statutory deadline or decides redress.
4. **Proportionality.** 12 of 15 source columns are loaded; ZIP code, company public response and send date are excluded; no narrative text or individual-level linkage.

## Fairness objectives
* F1 Tags (Older American, Servicemember) are audit-only, never features.
* F2 True-positive-rate and false-positive-rate gaps <= 0.05 across tag groups.
* F3 Proxy test: tags must not be recoverable from model features beyond chance.
* F4 No complaint is ever de-prioritised below its statutory response window.

## Review log
| Date | Gate | Outcome |
|------|------|---------|
| 2026-11-01 | Charter drafted (Module 2) | Pending panel review |
