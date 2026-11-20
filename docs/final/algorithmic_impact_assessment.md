# Algorithmic Impact Assessment

System: complaint service-failure risk model | Version: relief model `lightgbm@2026-11-01` (registry `cfpb-monetary_relief-risk` v1, alias `champion`)
Assessor: project author | Reviewer: **none yet (panel pending)** | Date: 2026-11-20 | Next review: with the first retraining or in 6 months

## 1. What the system does

Scores each new complaint with the calibrated probability that the company will close it with monetary relief, flags it when the score reaches its
region's review line (20% review capacity), and gives three reasons. A second, advisory model scores the chance of a late response. People read the flagged complaints first.

## 2. Who is affected

| Group | How | Interest |
|---|---|---|
| Complainants | Their complaint is read sooner or later | Fair, timely handling regardless of region, age or military status |
| Older Americans and Servicemembers (protected tags) | Audited for unequal flagging and unequal errors | Not under-served |
| Complaint reviewers and team leads | Use scores to choose what to read | Workable queue; no blame for overrides |
| Companies complained about | Their complaints are scored at different average levels | Not penalised for being complained about; company identity is the strongest input |
| Service quality lead, DPO, ethics panel | Own, approve, oversee | Defensible record |
| Regulator and public | Rely on the institution's claims | Honest account of limits |

## 3. Benefit

Reading the top 20% of complaints by score finds 52.1% of relief cases (2.6 times a random pick). At the deployed review lines 19.5% of complaints are flagged,
29.5% of flags end in relief (the base rate is 11.3%) and 50.9% of relief cases are caught. About half of relief cases are still missed.

## 4. Harms considered

Likelihood (L) and severity (S): Low, Medium, High. Residual is after the controls in column 5.

| # | Harm | L | S | Controls in place | Residual |
|---|---|---|---|---|---|
| H1 | Complaints of some regions or tag groups are read later because of the model | M | H | Per-region review lines (selection ratio 0.497 to 0.875); monthly selection-ratio alert (< 0.80); F2 tag-gap monitoring with intervals; F4 (no statutory deadline moves) | Medium: Older American TPR gap 0.047 with interval to 0.075 is unconfirmed; four-group comparison fails (0.061) |
| H2 | Tags are inferred from inputs and used implicitly (proxy) | M | H | F3 test at each retrain; company identified as the carrier; tags never inputs | **High**: F3 not met (AUC 0.680 and 0.640) |
| H3 | Complaints about new or small companies are mis-ranked | H | M | `unseen_values` returned on every score; unseen-share alert (warn 5%, hard 10%); route unseen to manual review | Medium: unseen-company ROC-AUC 0.626 |
| H4 | Reviewers accept scores uncritically (automation bias) | M | M | Reasons shown; override reason logged; override-rate alert; flag is a reading order only | Medium: not measured (replay only) |
| H5 | Scores used to delay complainants or to rank staff | M | H | Acceptable-use rule; F4; API returns no decision or deadline; access log | Medium until an acceptable-use policy is signed |
| H6 | Model learns company policy as if it were harm (label bias) | H | M | Label caveat in model card; ablation shows skill is company identity; no individual-level claim | High, and not removable with this data |
| H7 | Drift makes scores wrong without notice | H | M | Monthly PSI, Evidently drift, calibration and AUC alerts; retraining triggers; gates | Low to medium |
| H8 | Re-identification from stored requests | L | H | Requests hold only the nine intake fields; 90-day retention; no names, accounts, narratives, ZIP; k-anonymity k >= 5 upstream | Low |
| H9 | U.S. patterns assumed to hold in Ghana | H | M | Methodology demonstration only; local validation on approved, anonymised records required before any use | High until validated |
| H10 | Explanation misread as cause | M | M | Dashboards and API text state scores are associations; reasons are factors that moved the score | Medium |

## 5. Alternatives considered

* **Read in arrival order (status quo):** no model risk, but reviewers find relief cases at the base rate (11.3%).
* **Logistic regression:** statistically as good or slightly better and easier to explain. Not chosen because the tree model won on the validation window, where the model was selected, before the test window was opened; the paired test-window comparison came afterwards and favours the baseline. **A reviewer could reasonably prefer it**, and it is the right fallback.
* **Drop company identity:** removes the proxy carrier but loses most of the skill (ROC-AUC 0.638). Not workable as a prioritiser.

## 6. Decision

**Do not deploy operationally.** Acceptable as a research prototype and methodology demonstration on public data. Before any operational use:
(1) the panel reviews this assessment and records a decision; (2) local, anonymised bank data validate the model and rerun F2 and F3; (3) an acceptable-use policy is signed;
(4) the logistic baseline is re-evaluated as the primary model if F3 cannot be met; (5) a pilot measures override rate and reading-time effect (goal G5, which this project descoped).

## 7. Monitoring and review

Signals and thresholds: `fairness_monitoring_plan.md`. Incidents: `incident_response_plan.md`. This assessment is repeated at each retraining and at least every 6 months.
