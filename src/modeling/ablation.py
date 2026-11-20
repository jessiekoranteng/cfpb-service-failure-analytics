"""Does the model need company identity? Retrain without it and see what changes (Module 1 risk R1, Module 2 risk R-01).

    python -m src.modeling.ablation

Company identity carries 64.8% of SHAP importance and is the main carrier of the tag proxy signal (Module 4). Module 1 promised
to "train with and without Company"; this was not done in Module 4. Three variants share the champion's hyper-parameters and the
split, are calibrated on the validation window, and are scored once on the test window at the same rule (flag the top 20%
of the validation score distribution, one line for everyone, so the variants are comparable):

  full            all nine inputs (the deployed model, but with one line instead of region lines)
  no_company      company name and its recent volume replaced by a constant
  no_company_vol  as above, and also sub-issue (the next strongest driver)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling import config as C
from src.modeling import data as D
from src.monitoring import metrics as MM

VARIANTS = {"full": [], "no_company": ["company_key", "company_volume_lag4"], "no_company_or_subissue": ["company_key", "company_volume_lag4", "Sub-issue"]}


def neutralise(X: pd.DataFrame, cols: list[str], like: pd.DataFrame) -> pd.DataFrame:
    """Constant column (training mode, or NaN for company volume), so the input carries no information."""
    X = X.copy()
    for c in cols:
        X[c] = np.nan if c == "company_volume_lag4" else like[c].mode().iloc[0]
    return X


def run() -> dict:
    a, g = D.load()
    parts = {k: D.with_groups(v, g) for k, v in D.split(D.prepare(a), C.PRIMARY).items()}
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    champ = joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")
    y_tr, y_va, y_te = (p[C.PRIMARY].astype(int).to_numpy() for p in (tr, va, te))
    out = {}
    for name, cols in VARIANTS.items():
        Xtr, Xva, Xte = (neutralise(p[C.RAW_INPUT], cols, tr[C.RAW_INPUT]) for p in (tr, va, te))
        base = clone(champ["base"]).fit(Xtr, y_tr)
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(Xva, y_va)
        s_va, s = cal.predict_proba(Xva)[:, 1], cal.predict_proba(Xte)[:, 1]
        cut = float(np.quantile(s_va, 1 - C.REVIEW_CAPACITY))
        flag = (s >= cut).astype(int)
        r = {"removed": cols, "roc_auc": float(roc_auc_score(y_te, s)), "pr_auc": float(average_precision_score(y_te, s)),
             "pr_auc_lift": float(average_precision_score(y_te, s) / y_te.mean()), "flag_share": float(flag.mean()),
             "recall_at_flag": float(flag[y_te == 1].mean()), "precision_at_flag": float(y_te[flag == 1].mean()),
             "region_selection_ratio": MM.selection_ratio(flag, te["region"].astype(str))}
        for tag, col in (("older_american", "tag_older"), ("servicemember", "tag_servicemember")):
            gaps = MM.vs_rest_gaps(y_te, flag, te[col].astype(bool).to_numpy())
            r[f"f2_{tag}"] = {"tpr_gap": gaps["tpr_gap"], "fpr_gap": gaps["fpr_gap"]}
        out[name] = r
        print(f"{name:24s} ROC-AUC {r['roc_auc']:.3f} PR-AUC {r['pr_auc']:.3f} recall {r['recall_at_flag']:.3f} region ratio {r['region_selection_ratio']:.3f} "
              f"F2 older {r['f2_older_american']['tpr_gap']:.3f}/{r['f2_older_american']['fpr_gap']:.3f} serv {r['f2_servicemember']['tpr_gap']:.3f}/{r['f2_servicemember']['fpr_gap']:.3f}")
    path = C.ROOT / "reports" / "final" / "ablation_company.json"
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
