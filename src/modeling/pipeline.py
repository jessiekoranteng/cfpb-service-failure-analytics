"""scikit-learn pipelines shared by training, explainability and the API."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.modeling import config as C


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw inputs -> model features. Uses only values known at intake; no look-ahead."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = pd.DataFrame(X).copy()
        lag = pd.to_numeric(d["company_volume_lag4"], errors="coerce")
        d["company_volume_lag4"] = lag.astype("float32")
        d["company_volume_lag4_missing"] = lag.isna().astype("int8")
        d["log_company_volume_lag4"] = np.log1p(lag.fillna(0).clip(lower=0)).astype("float32")
        d["subissue_unknown"] = (d["Sub-issue"].astype(str) == "Unknown").astype("int8")
        for c in C.CATEGORICAL:
            d[c] = d[c].astype(object).where(d[c].notna(), "Unknown")
        return d[C.FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(C.FEATURES)


class CategoryCoder(BaseEstimator, TransformerMixin):
    """Fix the category set at training time so LightGBM sees the same codes at inference (unseen -> missing)."""

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(pd.Series(X[c]).dropna().astype(str).unique()) for c in C.CATEGORICAL}
        return self

    def transform(self, X):
        d = X.copy()
        for c in C.CATEGORICAL:
            d[c] = pd.Categorical(d[c].astype(str), categories=self.categories_[c])
        return d

    def get_feature_names_out(self, input_features=None):
        return np.array(C.FEATURES)


def _onehot_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=True), C.CATEGORICAL),
        ("num", StandardScaler(), ["log_company_volume_lag4"]),
        ("pass", "passthrough", ["company_volume_lag4_missing", "subissue_unknown"]),
    ], sparse_threshold=1.0)


def _ordinal_prep():
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), C.CATEGORICAL),
        ("num", "passthrough", ["company_volume_lag4", "month", "log_company_volume_lag4", "company_volume_lag4_missing", "subissue_unknown"]),
    ])


def make_baseline() -> Pipeline:
    return Pipeline([("features", FeatureBuilder()), ("clf", DummyClassifier(strategy="prior"))])


def make_logreg(**kw) -> Pipeline:
    params = {"C": 1.0, "class_weight": None, "max_iter": 300, "solver": "lbfgs"}
    params.update(kw)
    return Pipeline([("features", FeatureBuilder()), ("prep", _onehot_prep()), ("clf", LogisticRegression(**params))])


def make_forest(**kw) -> Pipeline:
    params = {"n_estimators": 80, "max_depth": 14, "min_samples_leaf": 50, "max_features": "sqrt", "class_weight": None, "n_jobs": 2,
              "random_state": C.SEED}
    params.update(kw)
    return Pipeline([("features", FeatureBuilder()), ("prep", _ordinal_prep()), ("clf", RandomForestClassifier(**params))])


def make_lgbm(**kw) -> Pipeline:
    from lightgbm import LGBMClassifier
    params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1,
              "colsample_bytree": 0.8, "reg_lambda": 1.0, "cat_smooth": 50, "min_data_per_group": 100, "n_jobs": 2,
              "random_state": C.SEED, "verbose": -1}
    params.update(kw)
    return Pipeline([("features", FeatureBuilder()), ("coder", CategoryCoder()), ("clf", LGBMClassifier(**params))])


MAKERS = {"majority": make_baseline, "logreg": make_logreg, "forest": make_forest, "lightgbm": make_lgbm}
