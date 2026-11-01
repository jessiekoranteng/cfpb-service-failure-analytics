"""Builds and executes notebooks/shap_analysis.ipynb (SHAP global/local, LIME, counterfactuals)."""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Module 4: SHAP, LIME and counterfactual analysis\n\n**Model:** champion monetary-relief risk model (LightGBM, isotonic-calibrated), scored on the **test window** (complaints received 2025-10 to 2026-06).\n"
       "SHAP values are on the base model's log-odds scale. Calibration is monotonic, so the ranking of complaints and the direction of every effect are unchanged.\n\n"
       "Explanations describe what the model associates with relief in historical complaints. They are not causes."),
    code("""import sys, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from IPython.display import display, Image
from src.modeling import config as C, data as D, explain as E
OUT = C.REPORTS
a, g = D.load(); a = D.prepare(a)
sp = D.split(a, C.PRIMARY)
te = D.with_groups(sp['test'], g).reset_index(drop=True); tr = sp['train']
bundle = E.load_bundle()
base, cal, thr = bundle['base'], bundle['calibrated'], bundle['review_threshold']
rng = np.random.default_rng(C.SEED)
idx = rng.choice(len(te), 6000, replace=False)
S = te.iloc[idx].reset_index(drop=True)
Xt, sv, base_value, names = E.shap_values(base, S[C.RAW_INPUT])
print(len(S), 'test complaints explained; review threshold', round(thr, 3))"""),
    md("## 1. Global feature importance (SHAP)"),
    code("""imp = E.global_importance(sv, names)
imp['mean_abs_shap'] = imp['mean_abs_shap'].round(4); imp['share'] = (imp['share']*100).round(1)
imp.to_csv(OUT/'shap_global_importance.csv', index=False)
E.plot_global(sv, Xt, names, OUT/'shap_bar.png', OUT/'shap_beeswarm.png')
display(imp[['label','mean_abs_shap','share']].rename(columns={'label':'Feature','mean_abs_shap':'Mean |SHAP|','share':'Share of total (%)'}))
display(Image(str(OUT/'shap_bar.png'))); display(Image(str(OUT/'shap_beeswarm.png')))"""),
    md("**Business reading.** The table shows which intake facts move the relief score most. Company and Issue/Sub-issue dominate: relief depends mostly on who the complaint is about and what went wrong, not on when or where it arrived."),
    code("""# Direction of effect for the most important categorical drivers: average SHAP by level
rows = []
for col in ['Issue', 'Submitted via', 'region', 'Product']:
    j = names.index(col)
    t = pd.DataFrame({'level': S[col].astype(str), 'shap': sv[:, j]}).groupby('level').agg(n=('shap','size'), mean_shap=('shap','mean'))
    t = t[t.n >= 50].sort_values('mean_shap')
    rows.append((col, pd.concat([t.head(2), t.tail(2)]).round(3)))
for col, t in rows:
    print(col); display(t)"""),
    md("## 2. Local explanations for individual complaints (SHAP waterfall)\nThree complaints from the test window: a correctly flagged relief case, a missed relief case, and a low-risk complaint."),
    code("""s_all = cal.predict_proba(S[C.RAW_INPUT])[:, 1]
S['score'] = s_all
y = S[C.PRIMARY].to_numpy()
pick = {
  'A. flagged and got relief (true positive)': int(np.argmax(np.where(y == 1, s_all, -1))),
  'B. relief missed (false negative)': int(np.argmax(np.where((y == 1) & (s_all < thr), s_all, -1))),
  'C. low risk, no relief': int(np.argmin(np.where(y == 0, s_all, 9))),
}
cases = []
for k, (label, i) in enumerate(pick.items()):
    xr = Xt.iloc[i]
    E.plot_waterfall(sv[i], base_value, xr, names, OUT/f'shap_waterfall_{k}.png', f'{label}: calibrated risk {s_all[i]:.2f}')
    contrib = E.top_contributions(sv[i], xr, names)
    cases.append({'case': label, 'row': i, 'score': round(float(s_all[i]), 3), 'flagged': bool(s_all[i] >= thr), 'actual_relief': int(y[i]), 'top_contributions': contrib})
    display(Image(str(OUT/f'shap_waterfall_{k}.png')))
(OUT/'local_cases.json').write_text(json.dumps(cases, indent=2, default=str))
pd.DataFrame([{k: v for k, v in c.items() if k != 'top_contributions'} for c in cases])"""),
    md("## 3. LIME (second opinion on the same three complaints)\nLIME fits a small linear model around each complaint. Agreement with SHAP on the main drivers is evidence the explanation is stable."),
    code("""rows = S.iloc[[pick[k] for k in pick]][C.RAW_INPUT]
exps = E.lime_explain(cal, tr[C.RAW_INPUT], rows)
lime_out = []
for (label, i), ex in zip(pick.items(), exps):
    items = ex.as_list()
    lime_out.append({'case': label, 'lime': [(f, round(w, 3)) for f, w in items]})
    fig = ex.as_pyplot_figure(); fig.suptitle(label, fontsize=9); fig.tight_layout()
    fig.savefig(OUT/f"lime_{label[0]}.png", dpi=130); plt.show()
(OUT/'lime_cases.json').write_text(json.dumps(lime_out, indent=2))
for o in lime_out: print(o['case'], o['lime'][:4])"""),
    md("## 4. Counterfactual explanations (DiCE)\n*What would need to change for a different review decision?* Only the two fields a handler records at intake (sub-issue and channel) may vary, and only to values that occur with the same issue; company, product, issue, region, month and company volume are held fixed so the alternatives stay realistic. The model is rescaled so that 0.5 equals the review threshold."),
    code("""rs = E.ReviewScale(cal, thr)
# queries: A (flagged), B (missed relief), plus the unflagged complaint closest to the review line, plus C (very low risk)
near = S[(S['score'] < thr)].assign(gap=lambda d: thr - d['score']).sort_values('gap')
near = near[near['company_key'] != 'BLOCK INC'].index[0]
qi = {'A (flagged)': pick['A. flagged and got relief (true positive)'], 'B (relief missed)': pick['B. relief missed (false negative)'],
      'D (just below the line)': int(near), 'C (very low risk)': pick['C. low risk, no relief']}
q = S.iloc[list(qi.values())]
res = E.counterfactuals(rs, tr, q, C.PRIMARY, n_cf=3)
cf_out = []
for (label, i), r in zip(qi.items(), res):
    orig = S.iloc[[i]][C.RAW_INPUT].reset_index(drop=True)
    row = {'case': label, 'company': str(orig['company_key'].iloc[0]), 'issue': str(orig['Issue'].iloc[0]), 'score': round(float(S.iloc[i]['score']), 3), 'flagged': bool(S.iloc[i]['score'] >= thr), 'changes': []}
    if isinstance(r, pd.DataFrame) and len(r):
        for _, cf in r.iterrows():
            row['changes'].append({c: (str(orig[c].iloc[0]), str(cf[c])) for c in C.RAW_INPUT if str(orig[c].iloc[0]) != str(cf[c])})
    else:
        row['note'] = 'no counterfactual found inside the plausible range'
    cf_out.append(row)
    print(label, '| company:', row['company'], '| score', row['score'], '| flagged', row['flagged'])
    for ch in row['changes']: print('    would flip with:', ch)
    if not row['changes']: print('    no plausible change flips the decision')
(OUT/'counterfactuals.json').write_text(json.dumps(cf_out, indent=2, default=str))"""),
    md("**Reading counterfactuals.** They show associations: a complaint recorded under a different issue or channel would be scored differently. They are a transparency tool for reviewers, not advice to change how a complaint is filed."),
]
ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
nbf.write(nb, ROOT / "notebooks" / "shap_analysis.ipynb")
print("notebook written")
