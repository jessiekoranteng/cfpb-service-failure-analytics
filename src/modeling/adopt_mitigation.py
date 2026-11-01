"""Adopt the equal-opportunity region thresholds (fitted on the validation window) into the deployed relief model bundle.

    python -m src.modeling.adopt_mitigation
"""
import json

import joblib
import mlflow

from src.modeling import config as C

if __name__ == "__main__":
    a = json.loads((C.REPORTS / f"analysis_{C.PRIMARY}.json").read_text())["mitigation_equal_opportunity"]
    path = C.MODELS / f"risk_{C.PRIMARY}.joblib"
    b = joblib.load(path)
    b["region_thresholds"] = a["thresholds"]
    joblib.dump(b, path)
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment(C.EXPERIMENT)
    with mlflow.start_run(run_name=f"{C.PRIMARY}-adopt-region-thresholds"):
        mlflow.set_tags({"target": C.PRIMARY, "stage": "mitigation-adopted"})
        mlflow.log_params({f"threshold_{k}": round(v, 4) for k, v in a["thresholds"].items()})
        mlflow.log_metrics({"region_selection_ratio": a["selection_ratio"], "region_fpr_gap": a["fpr_gap"], "region_tpr_gap": a["tpr_gap"],
                            "flag_share": a["flag_share"], "precision": a["precision"], "recall": a["recall"]})
        mlflow.log_artifact(str(path))
    print("adopted", a["thresholds"])
