"""Check the fairness objectives of the Ethical AI Charter (Modules 1-2, ETHICS_CHARTER.md) on the DEPLOYED relief model.

F2  true-positive-rate and false-positive-rate gaps <= 0.05 across tag groups
F3  proxy test: the protected tags must not be recoverable from the model's features beyond chance
F4  no complaint is de-prioritised below its statutory window (a design property, checked by the API contract test)

The deployed model flags a complaint when its calibrated score reaches the threshold for its region (bundle["region_thresholds"]);
this module therefore evaluates that configuration, and the single global threshold for comparison. Gaps get bootstrap
confidence intervals because a 0.05 limit is small next to the sampling error of groups with a few thousand rows.

    python -m src.modeling.charter
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.modeling import config as C
from src.modeling import data as D
from src.modeling import fairness as F
from src.modeling import pipeline as P

F2_LIMIT = 0.05
N_BOOT = 300


def deployed_flags(bundle, df, scores):
    """Flag when the calibrated score reaches the region's threshold (the API's rule); global threshold if the region is unknown."""
    cut = df["region"].astype(str).map(bundle.get("region_thresholds", {})).fillna(bundle["review_threshold"]).to_numpy(dtype=float)
    return (np.asarray(scores) >= cut).astype(int)


def _rates(y, f, grp, k):
    """Per-group true- and false-positive rates with bincount (fast enough to bootstrap)."""
    pos = y == 1
    npos = np.bincount(grp[pos], minlength=k)
    nneg = np.bincount(grp[~pos], minlength=k)
    tpr = np.bincount(grp[pos], weights=f[pos], minlength=k) / np.maximum(npos, 1)
    fpr = np.bincount(grp[~pos], weights=f[~pos], minlength=k) / np.maximum(nneg, 1)
    return tpr, fpr, npos > 0, nneg > 0


def _gaps(y, f, tg, to, ts, k):
    """TPR / FPR gaps: max-min over the four tag groups (tg holds integer codes), and group-vs-rest for each binary tag."""
    tpr, fpr, hp, hn = _rates(y, f, tg, k)
    out = {"four_group": {"tpr_gap": float(tpr[hp].max() - tpr[hp].min()), "fpr_gap": float(fpr[hn].max() - fpr[hn].min())}}
    for name, flag in (("older_american", to), ("servicemember", ts)):
        t, fp, _hp, _hn = _rates(y, f, flag, 2)
        out[name] = {"tpr_gap": float(abs(t[1] - t[0])), "fpr_gap": float(abs(fp[1] - fp[0]))}
    return out


def f2_check(y, flags, g, n_boot=N_BOOT, seed=C.SEED):
    """Point gaps and 95% bootstrap CIs. 'holds' = point estimate within the limit; 'holds_with_confidence' = the CI upper bound is too."""
    codes, labels = pd.factorize(g["tag_group"].astype(str))
    k = len(labels)
    to, ts = g["tag_older"].to_numpy().astype(int), g["tag_servicemember"].to_numpy().astype(int)
    point = _gaps(y, flags, codes, to, ts, k)
    rep = F.audit(y, flags, g)  # Fairlearn point estimates: the report tables, and a cross-check of the fast path
    assert abs(point["four_group"]["tpr_gap"] - rep["tag_group"]["tpr_gap"]) < 1e-3, "fast and Fairlearn gaps disagree"
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {kk: {"tpr_gap": [], "fpr_gap": []} for kk in point}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d = _gaps(y[i], flags[i], codes[i], to[i], ts[i], k)
        for kk in d:
            for m in d[kk]:
                draws[kk][m].append(d[kk][m])
    res = {}
    for kk in point:
        res[kk] = {}
        for m in ("tpr_gap", "fpr_gap"):
            lo, hi = np.percentile(draws[kk][m], [2.5, 97.5])
            res[kk][m] = {"gap": point[kk][m], "ci95": [float(lo), float(hi)]}
        res[kk]["holds"] = bool(max(point[kk]["tpr_gap"], point[kk]["fpr_gap"]) <= F2_LIMIT)
        res[kk]["holds_with_confidence"] = bool(max(res[kk]["tpr_gap"]["ci95"][1], res[kk]["fpr_gap"]["ci95"][1]) <= F2_LIMIT)
    return res, rep


def proxy_test(parts, g_train, g_test, bundle, n_train=150_000, n_boot=200, seed=C.SEED):
    """F3: train a classifier on the model's own nine inputs to predict each tag; AUC 0.5 = not recoverable.
    Also report how well the relief score itself separates tag from non-tag."""
    rng = np.random.default_rng(seed)
    tr, te = parts["train"], parts["test"]
    idx = rng.choice(len(tr), size=min(n_train, len(tr)), replace=False)
    Xtr, Xte = tr.iloc[idx][C.RAW_INPUT], te[C.RAW_INPUT]
    score = bundle["calibrated"].predict_proba(Xte)[:, 1]
    out = {}
    for name, col in C.PROTECTED.items():
        ytr = g_train[col].to_numpy()[idx]
        yte = g_test[col].to_numpy()
        pipe = P.make_lgbm(n_estimators=200, learning_rate=0.05)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if yte[i].min() != yte[i].max():
                boots.append(roc_auc_score(yte[i], p[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = {"auc_recover_tag_from_features": float(auc), "ci95": [float(lo), float(hi)],
                     "auc_relief_score_vs_tag": float(roc_auc_score(yte, score)), "tag_prevalence_test": float(yte.mean())}
    return out


def proxy_drivers(parts, g_train, g_test, n_train=150_000, seed=C.SEED):
    """Which input carries the proxy signal? AUC of the tag classifier when one input is dropped, and when only that input is used.
    (Raw inputs only: derived features are rebuilt from them inside the pipeline.)"""
    rng = np.random.default_rng(seed)
    tr, te = parts["train"], parts["test"]
    idx = rng.choice(len(tr), size=min(n_train, len(tr)), replace=False)
    out = {}
    for name, col in C.PROTECTED.items():
        ytr, yte = g_train[col].to_numpy()[idx], g_test[col].to_numpy()
        rows = {}
        for f in C.RAW_INPUT:
            res = {}
            for mode, cols in (("without", [c for c in C.RAW_INPUT if c != f]), ("only", [f])):
                # the pipeline derives features from several raw inputs, so give it the full frame with the unused ones neutralised
                Xtr, Xte = tr.iloc[idx][C.RAW_INPUT].copy(), te[C.RAW_INPUT].copy()
                for c in C.RAW_INPUT:
                    if c not in cols:
                        Xtr[c], Xte[c] = _neutral(Xtr[c]), _neutral(Xte[c], like=Xtr[c])
                pipe = P.make_lgbm(n_estimators=120, learning_rate=0.08)
                pipe.fit(Xtr, ytr)
                res[mode] = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
            rows[f] = res
        out[name] = rows
    return out


def _neutral(col, like=None):
    """Constant column of the training mode (or NaN for numerics), i.e. the input carries no information."""
    ref = like if like is not None else col
    if pd.api.types.is_numeric_dtype(ref):
        return pd.Series(np.nan, index=col.index, dtype="float32") if ref.name == "company_volume_lag4" else pd.Series(int(ref.mode().iloc[0]), index=col.index)
    return pd.Series(ref.mode().iloc[0], index=col.index)


def main():
    a, g = D.load()
    parts = {k: D.with_groups(v, g) for k, v in D.split(D.prepare(a), C.PRIMARY).items()}
    te = parts["test"]
    bundle = joblib.load(C.MODELS / f"risk_{C.PRIMARY}.joblib")
    y = te[C.PRIMARY].to_numpy()
    s = bundle["calibrated"].predict_proba(te[C.RAW_INPUT])[:, 1]
    glob = (s >= bundle["review_threshold"]).astype(int)
    dep = deployed_flags(bundle, te, s)
    res = {"limit": F2_LIMIT, "n_test": len(y)}
    for label, flags in (("global_threshold", glob), ("deployed_region_thresholds", dep)):
        r, rep = f2_check(y, flags, te)
        res[label] = {"flag_share": float(flags.mean()), "f2": r, "tag_group_table": rep["tag_group"]["table"],
                      "binary": {n: {k: rep[n][k] for k in ("selection_group", "selection_rest", "tpr_group", "tpr_rest", "fpr_group", "fpr_rest",
                                                            "disparate_impact_ratio", "equalized_odds_difference", "demographic_parity_difference")}
                                 for n in C.PROTECTED}}
    res["f3_proxy"] = proxy_test(parts, parts["train"], te, bundle)
    (C.REPORTS / "charter_checks.json").write_text(json.dumps(res, indent=2))
    for lab in ("global_threshold", "deployed_region_thresholds"):
        print(lab, "flag share", round(res[lab]["flag_share"], 4))
        for k, v in res[lab]["f2"].items():
            print(f"  F2 {k:15s} TPR gap {v['tpr_gap']['gap']:.4f} {v['tpr_gap']['ci95']}  FPR gap {v['fpr_gap']['gap']:.4f} {v['fpr_gap']['ci95']}  holds={v['holds']} with_confidence={v['holds_with_confidence']}")
    print("F3", json.dumps(res["f3_proxy"], indent=1))
    return res


def drivers_main():
    a, g = D.load()
    parts = {k: D.with_groups(v, g) for k, v in D.split(D.prepare(a), C.PRIMARY).items()}
    out = proxy_drivers(parts, parts["train"], parts["test"])
    (C.REPORTS / "proxy_drivers.json").write_text(json.dumps(out, indent=2))
    for name, rows in out.items():
        print(name)
        for f, r in sorted(rows.items(), key=lambda kv: kv[1]["without"]):
            print(f"  {f:22s} AUC without {r['without']:.3f}   only {r['only']:.3f}")


if __name__ == "__main__":
    import sys
    drivers_main() if "--drivers" in sys.argv else main()
