"""Explainability: SHAP (global + local), LIME (local) and DiCE (counterfactuals) for the champion risk model."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.modeling import config as C

PLAIN = {"company_key": "Company", "Issue": "Issue", "Sub-issue": "Sub-issue", "Sub-product": "Sub-product", "Product": "Product",
         "Submitted via": "Channel", "region": "Region", "company_volume_lag4": "Company volume (prior 4 wks)",
         "log_company_volume_lag4": "Company volume, log", "month": "Month received", "subissue_unknown": "Sub-issue not given",
         "company_volume_lag4_missing": "Volume history missing"}


def load_bundle(target=C.PRIMARY):
    return joblib.load(C.MODELS / f"risk_{target}.joblib")


def split_pipeline(base):
    """(feature steps, final estimator) of the fitted base pipeline."""
    return base[:-1], base[-1]


def shap_values(base, X_raw: pd.DataFrame):
    pre, clf = split_pipeline(base)
    Xt = pre.transform(X_raw)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = list(C.FEATURES) if isinstance(Xt, pd.DataFrame) else list(pre.get_feature_names_out())
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    base_value = expl.expected_value
    base_value = float(np.ravel(base_value)[-1])
    return Xt, np.asarray(sv), base_value, names


def global_importance(sv, names) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp["label"] = imp["feature"].map(PLAIN).fillna(imp["feature"])
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.reset_index(drop=True)


def plot_global(sv, Xt, names, path_bar, path_bee):
    labels = [PLAIN.get(n, n) for n in names]
    plt.figure(figsize=(7, 4.2))
    shap.summary_plot(sv, Xt.assign() if isinstance(Xt, pd.DataFrame) else Xt, feature_names=labels, plot_type="bar", show=False, max_display=10)
    plt.title("Mean absolute SHAP value (log-odds of monetary relief)")
    plt.tight_layout()
    plt.savefig(path_bar, dpi=150)
    plt.close()
    X_num = Xt.copy()
    if isinstance(X_num, pd.DataFrame):
        for c in C.CATEGORICAL:
            X_num[c] = X_num[c].cat.codes
    plt.figure(figsize=(7.5, 4.6))
    shap.summary_plot(sv, X_num, feature_names=labels, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(path_bee, dpi=150)
    plt.close()


def top_contributions(sv_row, x_row: pd.Series, names, k=5):
    order = np.argsort(-np.abs(sv_row))[:k]
    return [{"feature": PLAIN.get(names[i], names[i]), "value": str(x_row[names[i]]), "shap": float(sv_row[i])} for i in order]


def plot_waterfall(sv_row, base_value, x_row, names, path, title):
    labels = [f"{PLAIN.get(n, n)} = {str(x_row[n])[:28]}" for n in names]
    ex = shap.Explanation(values=sv_row, base_values=base_value, data=np.array([str(x_row[n])[:28] for n in names], dtype=object), feature_names=labels)
    plt.figure(figsize=(8, 4.2))
    shap.plots.waterfall(ex, max_display=8, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ LIME
class _Codec:
    """Integer codes for categoricals so LIME can perturb them, and the inverse for predict_proba."""

    def __init__(self, ref: pd.DataFrame):
        self.cats = {c: sorted(ref[c].astype(str).unique()) for c in C.CATEGORICAL}

    def encode(self, df):
        out = np.zeros((len(df), len(C.RAW_INPUT)))
        for j, c in enumerate(C.RAW_INPUT):
            if c in self.cats:
                idx = {v: i for i, v in enumerate(self.cats[c])}
                out[:, j] = df[c].astype(str).map(idx).fillna(0).to_numpy()
            else:
                out[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        return out

    def decode(self, arr):
        d = pd.DataFrame(arr, columns=C.RAW_INPUT)
        for c in C.CATEGORICAL:
            cats = np.array(self.cats[c], dtype=object)
            d[c] = cats[np.clip(np.rint(d[c].to_numpy()).astype(int), 0, len(cats) - 1)]
        return d


def lime_explain(model, ref: pd.DataFrame, rows: pd.DataFrame, num_features=6, n_samples=2000):
    from lime.lime_tabular import LimeTabularExplainer
    codec = _Codec(ref)
    ref_s = ref.sample(min(len(ref), 5000), random_state=C.SEED)
    explainer = LimeTabularExplainer(codec.encode(ref_s), feature_names=C.RAW_INPUT, class_names=["no relief", "relief"],
                                     categorical_features=[C.RAW_INPUT.index(c) for c in C.CATEGORICAL],
                                     categorical_names={C.RAW_INPUT.index(c): codec.cats[c] for c in C.CATEGORICAL}, discretize_continuous=True,
                                     random_state=C.SEED)
    fn = lambda arr: model.predict_proba(codec.decode(arr))
    return [explainer.explain_instance(codec.encode(rows.iloc[[i]])[0], fn, num_features=num_features, num_samples=n_samples) for i in range(len(rows))]


# ------------------------------------------------------------------ Counterfactuals (DiCE)
ACTIONABLE = ["Sub-issue", "Submitted via"]


def plausible_values(train: pd.DataFrame, row: pd.Series, min_count=30) -> dict:
    """Only values seen at least `min_count` times with the same product and issue, so counterfactuals stay realistic."""
    same = train[(train["Product"] == row["Product"]) & (train["Issue"] == row["Issue"])]
    return {c: sorted(same[c].astype(str).value_counts().loc[lambda v: v >= min_count].index.tolist()) for c in ACTIONABLE}


def counterfactuals(model, train: pd.DataFrame, query: pd.DataFrame, target: str, n_cf=3, sample=20000):
    """DiCE (random search). Company, product, issue, region, month and volume are fixed; only the sub-issue and channel recorded
    at intake may change, and only to values that occur with the same issue. Returns one result per query row (None if no counterfactual)."""
    import dice_ml
    d = train[C.RAW_INPUT + [target]].sample(min(sample, len(train)), random_state=C.SEED).reset_index(drop=True)
    for c in C.CATEGORICAL:
        d[c] = d[c].astype(str)
    d["month"] = d["month"].astype(int)
    d["company_volume_lag4"] = d["company_volume_lag4"].astype(float)
    data = dice_ml.Data(dataframe=d, continuous_features=["company_volume_lag4", "month"], outcome_name=target)
    exp = dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="random")
    out = []
    for i in range(len(query)):
        q = query.iloc[[i]][C.RAW_INPUT].copy()
        for c in C.CATEGORICAL:
            q[c] = q[c].astype(str)
        q["month"] = q["month"].astype(int)
        q["company_volume_lag4"] = q["company_volume_lag4"].astype(float)
        pr = plausible_values(train, q.iloc[0])
        try:
            res = exp.generate_counterfactuals(q, total_CFs=n_cf, desired_class="opposite", features_to_vary=ACTIONABLE,
                                               permitted_range=pr, random_seed=C.SEED)
            out.append(res.cf_examples_list[0].final_cfs_df)
        except Exception as e:  # noqa: BLE001  DiCE raises when no counterfactual exists inside the permitted ranges
            out.append(str(e)[:120])
    return out


class ReviewScale:
    """Wrap a calibrated model so probability 0.5 corresponds to the review threshold. DiCE then searches for the smallest
    change that moves a complaint across the review line, not across an arbitrary 50% probability."""

    def __init__(self, model, threshold):
        self.model, self.thr = model, float(threshold)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X = pd.DataFrame(X).copy()  # DiCE hands over object-typed frames; restore the numeric inputs
        for c in C.NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        p = self.model.predict_proba(X[C.RAW_INPUT])[:, 1]
        q = np.where(p < self.thr, 0.5 * p / self.thr, 0.5 + 0.5 * (p - self.thr) / (1 - self.thr))
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
