#!/usr/bin/env bash
# Two-stage monitoring run. Stage 1 scores the reference and replay windows with the deployed model (model environment,
# pandas 3). Stage 2 runs Evidently and Fairlearn (monitoring environment, pandas 2) and writes the dashboard files.
#
#   MODEL_PY=/opt/model/bin/python MON_PY=/opt/mon/bin/python scripts/run_monitoring.sh [--store path/to/explanations.jsonl]
set -euo pipefail
MODEL_PY=${MODEL_PY:-python}
MON_PY=${MON_PY:-python}
SCORED=${SCORED:-data/processed/monitoring_scored.parquet}
OUT=${OUT:-monitoring_app/assets}
"$MODEL_PY" -m src.monitoring.export_scored --out "$SCORED"
"$MON_PY" -m src.monitoring.job --scored "$SCORED" --out "$OUT/monitoring_data.json" --html "$OUT/evidently_drift_report.html" "$@"
"$MODEL_PY" -m src.modeling.retrain decide --monitoring "$OUT/monitoring_data.json" --out "$OUT/retrain_decision.json"
