"""Render the Model Card and the Fairness Metrics Report from reports/m4/facts.json.

Every figure in the two documents is read from the facts file (itself computed from the saved run outputs), so the
documents cannot drift from the models. Re-run after any retrain:  python -m src.modeling.facts && python -m src.modeling.docs_gen
"""
import json

from src.modeling import config as C

REL, UNT = C.PRIMARY, C.SECONDARY
N_BOOT_TXT = "300"
F: dict = {}  # reports/m4/facts.json, loaded by load()
B: dict = {}  # Module 3 bias-suite output, loaded by load()


def load():
    """Read the inputs once; kept out of import time so the module can be imported (and linted, tested) without them."""
    global F, B
    F = json.loads((C.REPORTS / "facts.json").read_text())
    B = json.loads((C.ROOT / "docs" / "evidence" / "bias_report.json").read_text())


def p(x, d=1):
    return f"{100 * x:.{d}f}%"


def n3(x):
    return f"{x:.3f}"


def n4(x):
    return f"{x:.4f}"


def k(x):
    return f"{int(x):,}"


def _mark(ok):
    return "met" if ok else "**NOT MET**"


_PLAIN = {"company_key": "Company", "Sub-issue": "Sub-issue", "Issue": "Issue", "Sub-product": "Sub-product", "Product": "Product",
          "Submitted via": "Channel", "region": "Region", "company_volume_lag4": "Company volume", "month": "Month"}
_TAG = {"older_american": "Older American", "servicemember": "Servicemember", "four_group": "All four tag groups (incl. Both)"}


def f2_rows(cfg, with_label=True):
    """Charter F2 table rows for one configuration ('global_threshold' or 'deployed_region_thresholds')."""
    out = []
    for key in ("older_american", "servicemember", "four_group"):
        d = F["charter"][cfg]["f2"][key]
        verdict = "holds, confirmed" if d["holds_with_confidence"] else ("holds on the point estimate only" if d["holds"] else "**does not hold**")
        out.append(f"| {_TAG[key]} | {n3(d['tpr_gap']['gap'])} ({n3(d['tpr_gap']['ci95'][0])}-{n3(d['tpr_gap']['ci95'][1])}) | "
                   f"{n3(d['fpr_gap']['gap'])} ({n3(d['fpr_gap']['ci95'][0])}-{n3(d['fpr_gap']['ci95'][1])}) | {verdict} |")
    return "\n".join(out)


def f3_rows():
    out = []
    for name in ("older_american", "servicemember"):
        r = F["charter"]["f3_proxy"][name]
        drv = F.get("proxy_drivers", {}).get(name, {})
        top = ", ".join(f"{_PLAIN.get(f, f)} ({v['only']:.2f})" for f, v in sorted(drv.items(), key=lambda kv: -kv[1]["only"])[:3]) or "not computed"
        out.append(f"| {_TAG[name]} | {n3(r['auc_recover_tag_from_features'])} ({n3(r['ci95'][0])}-{n3(r['ci95'][1])}) | {n3(r['auc_relief_score_vs_tag'])} | {top} |")
    return "\n".join(out)


def criteria_rows():
    rows = ["| Criterion | Target | Observed on test window | Result |", "|---|---|---|---|"]
    for c in F["criteria"]:
        note = f" ({c['note']})" if c.get("note") else ""
        rows.append(f"| {c['criterion']} | {c['target']} | {c['observed']:.3f} | {_mark(c['met'])}{note} |")
    return "\n".join(rows)


def model_card():
    perf, unt = F["champion_calibrated_test"][REL], F["champion_calibrated_test"][UNT]
    sp = F["split"][REL]
    cmp_ = F["champion_vs_logreg"]
    sens = F["sensitivity"]
    shap = {r["label"]: r for r in F["shap_global"]}
    rt = F["relief_at_review_threshold"]
    cap = F["capacity"]
    ft = F["fairness_tags"]
    mit = F["mitigation_equal_opportunity"]
    bef = F["region_before"]
    dep = F["charter"]["deployed_region_thresholds"]
    glob = F["charter"]["global_threshold"]
    perturb = {r["scenario"]: r for r in sens["perturbations"]}
    boot = sens["bootstrap"]
    return f"""# Model Card: CFPB complaint service-failure risk models (v1)

*Format after Mitchell et al. (2019). Numbers are generated from `reports/m4/facts.json`; do not edit by hand.*

## 1. Model details

| Item | Value |
|---|---|
| Owner / author | Jessie Kojo Koranteng (NXU, BAN6800 capstone) |
| Models | **Relief-risk model** (G1: will the complaint close with monetary relief?) and **late-response model** (G2: will the company miss the 15-day window?) |
| Algorithm | LightGBM gradient-boosted trees, isotonic-calibrated; chosen on validation PR-AUC from logistic regression, random forest and LightGBM |
| Version | v1, MLflow registry names `cfpb-monetary_relief-risk` and `cfpb-untimely-risk`, alias `champion` |
| Serialized files | `models/risk_monetary_relief.joblib`, `models/risk_untimely.joblib` (bundle: calibrated model, base pipeline, feature list, thresholds) |
| Inputs (9) | Product, Sub-product, Issue, Sub-issue, Submitted via (channel), region (from state), company, company complaint volume in the previous 4 weeks, month received |
| Outputs | Calibrated probability of monetary relief, calibrated probability of a late response, a review flag, top-3 reasons (SHAP) |
| Data | CFPB open complaint data, pseudonymised through the Module 3 pipeline |

## 2. Intended use

* **Primary use:** decision support for a bank's complaints-quality team (the generic Ghanaian retail-bank framing in Modules 1-2). Rank newly received complaints so that the {p(C.REVIEW_CAPACITY, 0)} with the highest relief risk get an early second-line review.
* **Intended users:** complaints analysts and team leads. A person always decides.
* **Status:** research prototype. Not approved for production: the ethics-panel sign-off required by the charter is pending and charter objective F3 is not met (Section 7).
* **Out of scope:** pricing, credit decisions, staff performance ranking, deciding which complainants to delay or deprioritise, any use on individuals' creditworthiness, and any deployment on bank-internal data without re-validation (all data here are U.S. complaints; see Limitations).

## 3. Factors and protected attributes

* **Audited:** Older American tag, Servicemember tag (complainant-reported in the CFPB data) and U.S. region (Northeast, Midwest, South, West, Other).
* **Never model inputs.** Tags are held in a restricted audit table and joined only for evaluation (leakage guard `assert_no_leakage`).
* **Not auditable:** gender, race and ethnicity are not in the public data, so no claim is made about them.

## 4. Data

* **Source:** CFPB Consumer Complaint Database, products "Checking or savings account" and "Money transfer, virtual currency, or money service", received 2020-01-01 onward: {k(535601)} in-scope records after the Module 3 pipeline.
* **Time-aware split by received week:** train {k(sp['train']['n'])} (to 2025-03), validation {k(sp['valid']['n'])} (2025-04 to 2025-09), test {k(sp['test']['n'])} (2025-10 to 2026-06, scored once). Complaints received after 2026-06-30 are excluded because many are still open.
* **Label prevalence:** relief {p(sp['train']['prevalence'])} train, {p(sp['valid']['prevalence'])} validation, {p(sp['test']['prevalence'])} test. Late response {p(F['split'][UNT]['train']['prevalence'], 2)} train, {p(F['split'][UNT]['valid']['prevalence'], 2)} validation, {p(F['split'][UNT]['test']['prevalence'], 2)} test.
* **Leakage controls:** `Company response to consumer`, `Timely response?` and every other outcome-related column are in a forbidden list; the guard fails the run if one appears. Company volume uses a 4-week lag.

## 5. Performance (test window, calibrated champion)

| Metric | Relief-risk model | Late-response model |
|---|---|---|
| ROC-AUC | {n3(perf['roc_auc'])} (95% CI {n3(boot['roc_auc_ci'][0])}-{n3(boot['roc_auc_ci'][1])}) | {n3(unt['roc_auc'])} |
| PR-AUC (prevalence) | {n3(perf['pr_auc'])} ({n3(perf['prevalence'])}); lift {perf['pr_auc_lift']:.2f}x; 95% CI {n3(boot['pr_auc_ci'][0])}-{n3(boot['pr_auc_ci'][1])} | {n3(unt['pr_auc'])} ({n3(unt['prevalence'])}); lift {unt['pr_auc_lift']:.2f}x |
| Recall when exactly the top 20% by score are flagged | {n3(cap[REL]['recall_top20'])} | {n3(cap[UNT]['recall_top20'])} |
| Precision when exactly the top 20% are flagged | {n3(cap[REL]['precision_top20'])} | {n3(cap[UNT]['precision_top20'])} |
| Brier score | {n4(perf['brier'])} | {n4(unt['brier'])} |
| Expected calibration error | {n4(F['calibration'][REL]['ece_calibrated'])} (uncalibrated {n4(F['calibration'][REL]['ece_uncalibrated'])}) | {n4(F['calibration'][UNT]['ece_calibrated'])} (uncalibrated {n4(F['calibration'][UNT]['ece_uncalibrated'])}) |

At the deployed review threshold ({C.REVIEW_CAPACITY:.0%} capacity rule, threshold {F['deployed_threshold']:.3f}), the relief model flags {p(F['test_flag_share'])} of test complaints with precision {n3(rt['precision'])} and recall {n3(rt['recall'])}.

**Success criteria.** The first twelve rows were pre-registered before training (`docs/module4/success_criteria.md`); the Charter rows (F2, F3 from `ETHICS_CHARTER.md`, Modules 1-2) are stricter than my pre-registered fairness limits and were added afterwards, so they are reported separately and govern:

{criteria_rows()}

**Honest note on the missed criterion.** The relief model is not better than the logistic-regression baseline on the test window. The paired-bootstrap difference in PR-AUC is {cmp_['pr_auc']['difference']:+.4f} (95% CI {cmp_['pr_auc']['ci95'][0]:+.4f} to {cmp_['pr_auc']['ci95'][1]:+.4f}) and in recall at capacity {cmp_['recall_at_capacity']['difference']:+.4f} (CI {cmp_['recall_at_capacity']['ci95'][0]:+.4f} to {cmp_['recall_at_capacity']['ci95'][1]:+.4f}). LightGBM was selected because it led on validation PR-AUC (by 0.001, a tie); the test window was touched once and was not used to re-select. The gap is small in practice, and the logistic-regression run is kept in MLflow (run `{F['models'][REL]['logreg'].get('run_id', 'see experiment')}`) as the challenger, and the next scheduled retrain should repeat the selection.

## 6. Explainability

* **Global (SHAP, TreeExplainer):** company identity carries {shap['Company']['share']:.1f}% of mean absolute SHAP importance, then sub-issue ({shap['Sub-issue']['share']:.1f}%), prior-4-week company volume ({shap['Company volume (prior 4 wks)']['share']:.1f}%), month received ({shap['Month received']['share']:.1f}%) and issue ({shap['Issue']['share']:.1f}%). Channel and product matter little.
* **Local:** SHAP waterfalls and LIME explanations for three cases (`reports/m4/shap_waterfall_*.png`, `lime_*.png`); the API returns the top-3 reasons per request.
* **Counterfactuals (DiCE):** restricted to plausible changes of sub-issue and channel; one case has no plausible counterfactual (very low-risk company), reported as such.
* **Reading:** the model mainly learns *which company* tends to resolve complaints with relief. That is a historical pattern in company behaviour, not a property of the complainant, and must not be read as a cause.

## 7. Fairness

All results are for the **deployed configuration** (calibrated score compared with a per-region threshold; {p(dep['flag_share'])} of test complaints flagged) on the test window. Gender and race cannot be audited (not in the data).

| Group | Disparate impact (>= 0.80) | Equalized-odds difference (<= 0.10) | Demographic-parity difference (<= 0.10) |
|---|---|---|---|
| Older American (n = {k(ft['older_american']['n_group'])}) | {n3(dep['binary']['older_american']['disparate_impact_ratio'])} | {n3(dep['binary']['older_american']['equalized_odds_difference'])} | {n3(dep['binary']['older_american']['demographic_parity_difference'])} |
| Servicemember (n = {k(ft['servicemember']['n_group'])}) | {n3(dep['binary']['servicemember']['disparate_impact_ratio'])} | {n3(dep['binary']['servicemember']['equalized_odds_difference'])} | {n3(dep['binary']['servicemember']['demographic_parity_difference'])} |

These are the limits I pre-registered. The **Ethical AI Charter** (Modules 1-2) is stricter, and its result is mixed:

| Charter F2: gaps <= 0.05 (95% bootstrap CI) | TPR gap | FPR gap | Verdict |
|---|---|---|---|
{f2_rows("deployed_region_thresholds")}

| Charter F3: can the tag be recovered from the model's inputs? (chance AUC = 0.50) | AUC from the nine inputs (95% CI) | AUC of the relief score alone | Strongest single input (AUC) |
|---|---|---|---|
{f3_rows()}

**F3 is not met:** the nine inputs recover both tags well above chance, mainly through the company (complainant populations differ by company), then the channel for Older Americans and the region for Servicemembers. The risk score itself separates the tags only weakly, and error-rate gaps are small, so there is no evidence that the model *acts* on the tags, but the proxy path the charter warns about exists (RAID R-02). **F2** holds on the point estimate for the two single tags, is confirmed for Servicemember, but is not confirmed for Older American (the interval reaches {n3(max(dep['f2']['older_american']['tpr_gap']['ci95'][1], dep['f2']['older_american']['fpr_gap']['ci95'][1]))}) and does not hold across the four tag groups, driven by the small "Both" group. With the single global threshold the Older American recall gap was {n3(glob['f2']['older_american']['tpr_gap']['gap'])}, just over the limit. **F4** is a design property: the response contains a review flag and reasons only, never a decision or a deadline (contract test `test_api_never_issues_a_decision_or_touches_a_deadline`).

**Region (extra audit):** before mitigation the lowest-to-highest selection-rate ratio was {n3(bef['selection_ratio_min_max'])} and the true-positive-rate gap {n3(bef['tpr_gap'])} (Northeast flagged most, South least); part of this reflects real differences in relief rates by region. After per-region equal-opportunity thresholds fitted on validation data: ratio {n3(mit['selection_ratio'])}, true-positive-rate gap {n3(mit['tpr_gap'])}, false-positive-rate gap {n3(mit['fpr_gap'])}, flag share {p(mit['flag_share'])}, precision {n3(mit['precision'])}, recall {n3(mit['recall'])}. The thresholds are stored in the bundle and applied by the API. Full detail: `docs/module4/fairness_metrics_report.md`.

## 8. Sensitivity and robustness

* Bootstrap (300 resamples) 95% CI for ROC-AUC {n3(boot['roc_auc_ci'][0])}-{n3(boot['roc_auc_ci'][1])}.
* Company volume +/-20%: ROC-AUC {n3(perturb['company volume -20%']['roc_auc'])} / {n3(perturb['company volume +20%']['roc_auc'])} (baseline {n3(perturb['baseline']['roc_auc'])}); flag flips {p(perturb['company volume -20%']['flag_flip_rate'])} / {p(perturb['company volume +20%']['flag_flip_rate'])}.
* Sub-issue not recorded: ROC-AUC {n3(perturb['sub-issue not recorded']['roc_auc'])}, {p(perturb['sub-issue not recorded']['flag_flip_rate'])} of flags change.
* Company volume missing: ROC-AUC {n3(perturb['company volume missing']['roc_auc'])}, {p(perturb['company volume missing']['flag_flip_rate'])} of flags change.
* **Unseen company:** ROC-AUC falls to {n3(perturb['unseen company']['roc_auc'])} and PR-AUC to {n3(perturb['unseen company']['pr_auc'])}; {p(perturb['unseen company']['flag_flip_rate'])} of flags change. The API reports every unseen category in `unseen_values`.
* Quarterly ROC-AUC on test: {", ".join(f"{q['quarter']} {n3(q['roc_auc'])}" for q in sens['quarterly'])}.
* Removing the January 2025 complaint surge ({k(sens['without_surge']['rows_removed'])} rows) from training changes ROC-AUC from {n3(sens['without_surge']['with_surge']['roc_auc'])} to {n3(sens['without_surge']['without_surge']['roc_auc'])}.

## 9. Limitations and risks

1. **Jurisdiction:** all data are U.S. complaints; validity for a Ghanaian bank is untested (RAID R-06).
2. **Label bias:** monetary relief is a *company* decision; the model reproduces company practice, including any inconsistency (R-01).
3. **Company dominance:** about two thirds of importance sits in company identity; new entrants are scored poorly (see unseen-company result).
4. **Drift:** volume and relief rate moved sharply year to year (R-07). Calibration got slightly worse on the test window (ECE {n4(F['calibration'][REL]['ece_calibrated'])} vs {n4(F['calibration'][REL]['ece_uncalibrated'])} uncalibrated) because isotonic calibration was fitted on a validation window with a different relief rate.
5. **Late-response model:** the less stable of the two. Its validation-window ROC-AUC ({n3(F['models'][UNT]['lightgbm']['valid']['roc_auc'])}) differs from the test window ({n3(unt['roc_auc'])}), prevalence moved from {p(F['split'][UNT]['valid']['prevalence'], 2)} to {p(F['split'][UNT]['test']['prevalence'], 2)}, and the cut-off fixed on validation to flag 20% would flag {p(cap[UNT]['share_flagged_by_fixed_threshold'])} of test complaints because scores shifted with the lower test prevalence (exactly the top 20% by score would catch {p(cap[UNT]['recall_top20'], 0)} of late responses). Use its probability as advice only; the API's review flag comes from the relief model.
6. **Misuse (R-04) and automation bias (R-08):** scores must not be used to delay complainants or to rank staff; analysts must see the reasons and can override.
7. **Proxy risk (charter F3, not met):** the company (and, to a lesser degree, channel and region) lets a classifier recover the Older American and Servicemember tags (AUC {n3(F['charter']['f3_proxy']['older_american']['auc_recover_tag_from_features'])} and {n3(F['charter']['f3_proxy']['servicemember']['auc_recover_tag_from_features'])}). Company is also where most of the model's skill lives, so dropping it is not a fix; the risk is accepted for a prototype only and must be reviewed by the ethics panel before any use.
8. **Group sizes:** the Older American and Servicemember groups are 7-9% of test rows; the intersection ("Both", n = 1,349) is too small for stable estimates.

## 10. Acceptance criteria and monitoring (proposed)

| Signal | Cadence | Alert when |
|---|---|---|
| ROC-AUC, PR-AUC lift (relief) | Quarterly, once labels mature | ROC-AUC < 0.70 or lift < 2.0 |
| Recall at 20% capacity | Quarterly | < 0.45 |
| Expected calibration error | Monthly | > 0.03 |
| Selection-rate ratio by region and tag | Monthly | < 0.80 |
| TPR and FPR gaps by tag, with bootstrap interval | Monthly | point estimate > 0.05 (charter F2) |
| Tag recoverability from inputs (AUC), re-run at each retrain | Each retrain | CI excludes 0.50 (charter F3); escalate to the panel |
| True-positive-rate gap by region and tag | Quarterly | > 0.10 |
| Share of requests with `unseen_values` | Weekly | > 5% |
| Share flagged for review | Weekly | outside 15%-25% |
| Company-mix drift (population stability index) | Monthly | > 0.20 |
| Analyst override rate | Monthly | < 5% (automation bias) or > 60% (model not trusted) |

Retraining: quarterly or when two alerts coincide. Every retrain repeats selection on a fresh validation window, the paired comparison with logistic regression, and the full fairness audit.

## 11. Ethical considerations and caveats

The model supports a human review queue; it does not decide outcomes for individuals. Protected attributes are never inputs, but they are partly recoverable from the inputs that are used (Section 7). Fairness was audited only for attributes present in the data. Region disparity was found and mitigated at a measured cost (Section 7). Charter F3 is not met and F2 is not confirmed for Older Americans, so the model must not be used on live complaints before the ethics panel has reviewed this card. Because a reduction in one gap can widen another, any retrain must re-run the full audit.

## 12. Reproducibility

`python -m src.modeling.train`, `python -m src.modeling.analysis`, `python -m src.modeling.adopt_mitigation`, `python -m src.modeling.compare`, `python -m src.modeling.facts`; seed {C.SEED}; MLflow experiment `{C.EXPERIMENT}` ({F['mlflow']['runs_total']} runs, {F['mlflow']['tuning_trials_nested']} nested tuning trials).
"""


def fairness_report():
    ft = F["fairness_tags"]
    bef, ta, eo = F["region_before"], F["mitigation_threshold_optimizer"], F["mitigation_equal_opportunity"]
    tg = F["tag_group_before"]
    di_u = F["fairness_untimely_di"]
    dep = F["charter"]["deployed_region_thresholds"]
    glob = F["charter"]["global_threshold"]
    cov = {g["value"]: g for g in B["checks"]["coverage"]["groups"] if g["group"] == "Submitted via"}
    fax, email = cov["Fax"], cov["Email"]
    drift = {(g["group"], g["value"]): g["max_abs_share_change"] for g in B["checks"]["representation_drift"]["groups"]}
    web_drift, prod_drift = drift[("Submitted via", "Web")], drift[("Product", "Checking or savings account")]
    south_drift = drift[("region", "South")]
    miss_all = {c["by"]: c for c in B["checks"]["differential_missingness"]["checks"]}
    miss, miss_region = miss_all["Submitted via"], miss_all["region"]
    od = B["checks"]["outcome_disparity"]["labels"]["monetary_relief"]
    B_reg, B_tag = od["region"], od["tag_group"]
    reg_rates = {g["value"]: g["rate"] for g in B_reg["by_group"]}

    def tag_rows(block, label):
        out = []
        for g, name in (("older_american", "Older American"), ("servicemember", "Servicemember")):
            b = block[g]
            out.append(f"| {label} | {name} | {k(b['n_group'])} | {p(b['selection_group'])} / {p(b['selection_rest'])} | {n3(b['tpr_group'])} / {n3(b['tpr_rest'])} | {n3(b['fpr_group'])} / {n3(b['fpr_rest'])} | {n3(b['disparate_impact_ratio'])} | {n3(b['equalized_odds_difference'])} | {n3(b['demographic_parity_difference'])} |")
        return out

    region_rows = "\n".join(f"| {r['region']} | {k(r['n'])} | {p(r['selection'])} | {n3(r['tpr'])} | {n3(r['fpr'])} |" for r in bef["table"])
    eo_rows = "\n".join(f"| {r['region']} | {k(r['n'])} | {p(r['selection'])} | {n3(r['tpr'])} | {n3(r['fpr'])} | {eo['thresholds'][r['region']]:.3f} |" for r in eo["table"])
    tgrows = "\n".join(f"| {r['tag_group']} | {k(r['n'])} | {p(r['selection'])} | {n3(r['tpr'])} | {n3(r['fpr'])} |" for r in tg["table"])
    return f"""# Fairness Metrics Report: relief-risk model (v1)

*Generated from `reports/m4/facts.json`. Library: Fairlearn `MetricFrame`, `demographic_parity_difference/ratio`, `equalized_odds_difference`; disparate impact ratio = lower selection rate divided by higher selection rate, so it is below 1 whichever group is flagged more often. Test window 2025-10 to 2026-06, flags at the deployed threshold.*

## 1. Fairness objectives (from Modules 1-2)

The Module 1 Vision and Module 2 Project Overview commit the project to **fair prioritisation (G4)**: the review queue must not systematically under- or over-select complainants because of who they are. Fixed before training (`docs/module4/success_criteria.md`): disparate impact ratio >= 0.80, equalized-odds difference <= 0.10, demographic-parity difference <= 0.10, for each protected group against everyone else. If a criterion is missed: mitigate and report the accuracy cost.

**Protected attributes available:** the Older American tag (age 62+) and the Servicemember tag. Gender, race and ethnicity are absent from the public data and **cannot be audited**; no fairness claim is made about them. Region (U.S. Census region from the state code) was audited as an additional, non-pre-registered dimension because Module 3's bias suite could block on it.

## 2. Data bias carried in from Module 3

The Module 3 bias suite ran on {k(B['n_records'])} records with overall status **{B['overall']}**:

* **Coverage:** tags are recorded on only {p(1 - B['checks']['coverage']['groups'][0]['share'])} of complaints (the rest "None recorded"), so a tag audit has a recording-bias caveat. Fax (n = {k(fax['n'])}) and Email (n = {k(email['n'])}) are too small to audit.
* **Representation drift** across years: largest shifts in channel mix (Web {p(web_drift)} change in share) and product mix ({p(prod_drift)}), and a smaller shift for South ({p(south_drift)}).
* **Differential missingness:** the share of missing fields varies by channel from {p(miss['min_rate'])} to {p(miss['max_rate'])}, but not meaningfully by region (gap {p(miss_region['gap'])}).
* **Outcome disparity in the raw labels** (before any model): relief rates by region run from {p(reg_rates['Other'])} (Other) to {p(reg_rates['Northeast'])} (Northeast), a ratio of {n3(B_reg['ratio'])}; by tag group the ratio is {n3(B_tag['ratio'])}. These are real differences in the historical outcomes, so a model that predicts them well will select some groups more often than others. This is why region selection rates in Section 4 differ *before* any model bias is considered, and why equal-opportunity (equal recall among true positives) was preferred to equal selection rates.
* **Concentration:** the top five companies account for {p(B['checks']['concentration']['top5_share'])} of records ({k(B['checks']['concentration']['n_companies'])} companies, HHI {n3(B['checks']['concentration']['hhi'])}). Relief rates span 0%-33% among companies with 1,000+ complaints (RAID R-01), which is why company identity dominates the model.

## 3. Results for the pre-registered groups

| Stage | Group | n in group | Selection rate (group / rest) | TPR (group / rest) | FPR (group / rest) | Disparate impact | Equalized-odds diff | Demographic-parity diff |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(tag_rows(ft, "Before mitigation"))}
{chr(10).join(tag_rows(eo["tags"], "After region thresholds"))}

**Reading.** All pre-registered criteria hold before mitigation. Older Americans are flagged slightly *more* often ({p(ft['older_american']['selection_group'])} vs {p(ft['older_american']['selection_rest'])}) but caught slightly *less* often when relief was due (TPR {n3(ft['older_american']['tpr_group'])} vs {n3(ft['older_american']['tpr_rest'])}) and more often wrongly (FPR {n3(ft['older_american']['fpr_group'])} vs {n3(ft['older_american']['fpr_rest'])}): a small recall gap of {n3(ft['older_american']['equalized_odds_difference'])}, inside the limit but the one to watch. The late-response model gives disparate impact ratios of {n3(di_u['older_american'])} (Older American) and {n3(di_u['servicemember'])} (Servicemember); the Servicemember value is closest to the 0.80 line and is a monitoring item.

Combined tag groups (before mitigation):

| Tag group | n | Selection rate | TPR | FPR |
|---|---|---|---|---|
{tgrows}

Intersection "Both" is small (n = {k(tg['table'][0]['n'])}); estimates are unstable and are not used for decisions.

## 4. Region disparity: detected

| Region | n | Selection rate | TPR | FPR |
|---|---|---|---|---|
{region_rows}

Selection-rate ratio (lowest/highest) = **{n3(bef['selection_ratio_min_max'])}** and TPR gap = **{n3(bef['tpr_gap'])}**; both outside the limits used for the tags. The Northeast is flagged at {p(bef['table'][1]['selection'])} with recall {n3(bef['table'][1]['tpr'])}, against the South at {p(bef['table'][3]['selection'])} and {n3(bef['table'][3]['tpr'])}. Region is not a model input as a protected attribute, but the model receives it, and company mix differs by region (a proxy path, RAID R-02).

## 5. Mitigation options compared

| Option | TPR gap | FPR gap | Selection ratio | Share flagged | Precision | Recall |
|---|---|---|---|---|---|---|
| None | {n3(bef['tpr_gap'])} | {n3(bef['fpr_gap'])} | {n3(bef['selection_ratio_min_max'])} | {p(ta['before']['flag_share'])} | {n3(ta['before']['precision'])} | {n3(ta['before']['recall'])} |
| Fairlearn ThresholdOptimizer (equalized odds across region) | {n3(ta['after']['tpr_gap'])} | {n3(ta['after']['fpr_gap'])} | {n3(ta['after']['selection_ratio'])} | {p(ta['after']['flag_share'])} | {n3(ta['after']['precision'])} | {n3(ta['after']['recall'])} |
| **Per-region equal-opportunity thresholds (adopted)** | {n3(eo['tpr_gap'])} | {n3(eo['fpr_gap'])} | {n3(eo['selection_ratio'])} | {p(eo['flag_share'])} | {n3(eo['precision'])} | {n3(eo['recall'])} |

ThresholdOptimizer closes the selection-rate gap but raises the flagged share from {p(ta['before']['flag_share'])} to {p(ta['after']['flag_share'])}, which nearly doubles the review workload against a 20% capacity: **rejected as operationally infeasible**. The adopted method keeps the workload ({p(eo['flag_share'])}) and equalises recall across regions by choosing one threshold per region on the *validation* window (equal-opportunity target recall {n3(eo['recall_target'])}); thresholds are applied on the untouched test window.

Per-region equal-opportunity results:

| Region | n | Selection rate | TPR | FPR | Threshold |
|---|---|---|---|---|---|
{eo_rows}

**Cost and residual risk.** Overall precision changes from {n3(ta['before']['precision'])} to {n3(eo['precision'])} and recall from {n3(ta['before']['recall'])} to {n3(eo['recall'])}. The TPR gap falls from {n3(bef['tpr_gap'])} to {n3(eo['tpr_gap'])} (inside 0.10, but only just) and the selection ratio rises to {n3(eo['selection_ratio'])}. Tag results stay inside every limit after mitigation (Section 3). Residual risks: (1) "Other" has n = {k(eo['table'][2]['n'])}, so its threshold ({eo['thresholds']['Other']:.3f}) is noisy; (2) West still has the lowest recall ({n3(eo['table'][4]['tpr'])}); (3) per-group thresholds are a policy choice that treats groups differently on purpose, which must be approved and documented by governance before use.

## 6. Objectives of the Ethical AI Charter (Modules 1-2)

The charter (`ETHICS_CHARTER.md`) sets stricter fairness objectives than the limits I pre-registered. They govern, and I evaluated them after the test window had been scored (see the addendum in `docs/module4/success_criteria.md`); nothing was re-tuned in response.

**F2: true-positive-rate and false-positive-rate gaps <= 0.05 across tag groups** (gap, 95% bootstrap interval from {N_BOOT_TXT} resamples of the test window):

| Single global threshold | TPR gap | FPR gap | Verdict |
|---|---|---|---|
{f2_rows("global_threshold")}

| Deployed per-region thresholds | TPR gap | FPR gap | Verdict |
|---|---|---|---|
{f2_rows("deployed_region_thresholds")}

*Reading.* The Servicemember gaps are small and the limit is confirmed. For Older Americans the global threshold missed the limit by {n4(glob['f2']['older_american']['tpr_gap']['gap'] - 0.05)} and the deployed thresholds meet it on the point estimate, but the interval is wide (the group has {k(ft['older_american']['n_group'])} test rows), so **the 0.05 limit cannot be confirmed**. The four-group view fails ({n3(dep['f2']['four_group']['tpr_gap']['gap'])}) because the "Both" group (n = {k(dep['tag_group_table'][0]['n'])}) has a recall of {n3(dep['tag_group_table'][0]['tpr'])}, against {n3(dep['tag_group_table'][1]['tpr'])} for complaints with no tag. An honest reading is that gaps of this size are not distinguishable from noise with the available data, not that the model is fair at 0.05.

**F3: proxy test.** A LightGBM classifier trained on the model's own nine inputs to predict each tag (150,000 training complaints, scored on the test window):

| Tag | AUC from the nine inputs (95% CI) | AUC of the relief score alone | Strongest single input (AUC) |
|---|---|---|---|
{f3_rows()}

*Reading.* **F3 is not met.** Both tags are recoverable well above chance (0.50). The risk score alone carries much less tag information, which is why the error-rate audit above is mostly clean, but the inputs are proxies (RAID R-02) and the model could come to use them if the company mix changes. Removing the main proxy is not a fix: company identity carries about two thirds of the model's SHAP importance, so dropping it would remove most of the predictive skill.

**F4: no complaint is de-prioritised below its statutory window.** Design property. The API returns a review flag, probabilities and reasons only; there is no decision, deadline or ranking-below-window output, and a contract test fails if one is added.

**Recommendation.** (1) Treat the model as a research prototype; (2) ask the ethics panel to rule on whether F3 should be read as written or restated as a bound on how much the *score* depends on the tags; (3) monitor F2 monthly with its interval and escalate when the point estimate exceeds 0.05; (4) do not add tag-specific thresholds, which would use the tags as decision inputs and breach charter F1.

## 7. What this audit does not show

Tags are self-reported on a minority of complaints; absence of a tag is not absence of the characteristic. Equalised error rates do not make the *labels* fair: relief is a company decision, so the model can be perfectly consistent and still copy inconsistent company practice. No gender or race audit is possible. Fairness metrics are computed on a single test window; confidence intervals for the smaller groups are wide.
"""


def main():
    load()
    (C.ROOT / "MODEL_CARD.md").write_text(model_card())
    (C.ROOT / "docs" / "module4").mkdir(parents=True, exist_ok=True)
    (C.ROOT / "docs" / "module4" / "fairness_metrics_report.md").write_text(fairness_report())
    print("wrote MODEL_CARD.md and docs/module4/fairness_metrics_report.md")


if __name__ == "__main__":
    main()
