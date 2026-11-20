"""Monitoring statistics. Pure numpy/pandas so the same code runs in the model environment (pandas 3) and in the
Evidently environment (pandas 2). Nothing here reads files or knows about the model."""
import numpy as np
import pandas as pd

from src.monitoring.thresholds import MIN_CELL

EPS = 1e-4


def psi_categorical(ref: pd.Series, cur: pd.Series, top: int = 50) -> float:
    """Population stability index over the `top` most common reference categories plus 'other' (unseen values land in 'other')."""
    keep = ref.astype(str).value_counts().index[:top]
    r = ref.astype(str).where(ref.astype(str).isin(keep), "__other__").value_counts(normalize=True)
    c = cur.astype(str).where(cur.astype(str).isin(keep), "__other__").value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0).clip(lower=EPS), c.reindex(idx, fill_value=0).clip(lower=EPS)
    return float(((c - r) * np.log(c / r)).sum())


SCORE_EDGES = np.array([-np.inf, 0.02, 0.05, 0.08, 0.12, 0.17, 0.25, 0.40, np.inf])


def psi_numeric(ref, cur, edges=SCORE_EDGES) -> float:
    """PSI over fixed bins. Fixed edges, not reference deciles: the isotonic calibrator outputs a step function, so many
    reference scores tie and decile edges collapse (that artefact produced false 'hard' alerts in the first run)."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(((c - r) * np.log(c / r)).sum())


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error, identical to src/modeling/analysis.py (sklearn quantile bins, weighted by bin share)."""
    from sklearn.calibration import calibration_curve
    y, p = np.asarray(y), np.asarray(p)
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    w = np.histogram(p, bins=np.quantile(p, np.linspace(0, 1, bins + 1)))[0] / len(p)
    return float(np.sum(np.abs(frac - mean) * w[: len(frac)]))


def group_rates(y, flag, g):
    """Selection rate, TPR, FPR and counts for one boolean group mask. Rates are None when a cell is below MIN_CELL."""
    y, flag, g = np.asarray(y).astype(int), np.asarray(flag).astype(int), np.asarray(g, bool)
    pos, neg = g & (y == 1), g & (y == 0)
    return {"n": int(g.sum()), "positives": int(pos.sum()), "negatives": int(neg.sum()),
            "flag_rate": float(flag[g].mean()) if g.any() else None,
            "tpr": float(flag[pos].mean()) if pos.sum() >= MIN_CELL else None,
            "fpr": float(flag[neg].mean()) if neg.sum() >= MIN_CELL else None}


def vs_rest_gaps(y, flag, g) -> dict:
    """Charter F2: |TPR_group - TPR_rest| and |FPR_group - FPR_rest| (the 'group vs everyone else' definition used in Module 4)."""
    a, b = group_rates(y, flag, g), group_rates(y, flag, ~np.asarray(g, bool))
    gap = lambda k: None if a[k] is None or b[k] is None else abs(a[k] - b[k])
    return {"tpr_gap": gap("tpr"), "fpr_gap": gap("fpr"), "n_group": a["n"], "positives_group": a["positives"]}


def bootstrap_gaps(y, flag, g, n_boot: int = 300, seed: int = 42) -> dict:
    """95% interval for the two F2 gaps. The interval is what separates 'gap above the limit' from 'cannot tell'."""
    rng = np.random.default_rng(seed)
    y, flag, g = np.asarray(y), np.asarray(flag), np.asarray(g, bool)
    n = len(y)
    t, f = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r = vs_rest_gaps(y[i], flag[i], g[i])
        if r["tpr_gap"] is not None:
            t.append(r["tpr_gap"])
        if r["fpr_gap"] is not None:
            f.append(r["fpr_gap"])
    ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) > 20 else None
    return {"tpr_ci95": ci(t), "fpr_ci95": ci(f)}


def size_tiers(train_company: pd.Series, company: pd.Series) -> pd.Series:
    """Company size tier from complaint counts in the TRAINING window only (so tiers do not use future data).
    Top 5 | Large (6-50) | Mid (51-300) | Small (rest) | Unseen (not in training)."""
    rank = train_company.value_counts().rank(method="first", ascending=False)
    r = company.map(rank)
    out = pd.Series("Unseen", index=company.index, dtype=object)
    out[r <= 5] = "Top 5"
    out[(r > 5) & (r <= 50)] = "Large (6-50)"
    out[(r > 50) & (r <= 300)] = "Mid (51-300)"
    out[r > 300] = "Small (301+)"
    return out


def selection_ratio(flag, region) -> float | None:
    """Lowest regional flag rate divided by the highest (four-fifths screen)."""
    s = pd.Series(np.asarray(flag, float)).groupby(np.asarray(region)).mean()
    return float(s.min() / s.max()) if len(s) > 1 and s.max() > 0 else None
