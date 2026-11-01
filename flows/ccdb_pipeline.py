"""Prefect orchestration DAG for the CFPB service-failure data pipeline (Module 3).

DAG (each box is a Prefect task; arrows are data dependencies):

  ingest -> validate_raw -> clean -> transform -> integrate -> validate_curated -> bias_suite -> anonymise -> publish
                                                                  \\-> dashboard_feed ----------------------/

Failure policy: validate_raw / validate_curated fail the run on any *critical* expectation; the bias suite fails the
run only on a BLOCK severity; anonymise fails if any PII pattern remains. Network or I/O tasks retry twice.

Run locally:   python -m flows.ccdb_pipeline --source data/raw/complaints.csv.zip
Schedule it:   python -m flows.ccdb_pipeline --serve --cron "0 6 * * *"
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from src.ingestion.ccdb_ingest import ingest as ingest_source
from src.pipeline import anonymise as anon
from src.pipeline import bias as bias_mod
from src.pipeline import clean as clean_mod
from src.pipeline import integrate as integ
from src.pipeline import transform as trans
from src.pipeline import validate as val
from src.pipeline.audit import AuditLog, frame_fingerprint, verify_chain
from src.pipeline.config import Paths
from src.pipeline.governance import check_access, load_policy
from src.pipeline.lineage import Lineage, code_hash, file_sha256


@task(name="ingest", retries=2, retry_delay_seconds=10)
def t_ingest(source: str, paths: Paths, audit: AuditLog) -> pd.DataFrame:
    out = paths.interim / "scope.parquet"
    meta = ingest_source(source, out)
    df = pd.read_parquet(out)
    audit.event("ingest", "ccdb_source", "scope-filtered snapshot for analytics", rows_in=meta["source_rows"],
                rows_out=len(df), columns=df.columns, output_hash=file_sha256(out), detail={"source_sha256": meta["source_sha256"]})
    return df


@task(name="validate_raw")
def t_validate_raw(df: pd.DataFrame, paths: Paths, audit: AuditLog, min_rows: int) -> dict:
    summary = val.run_suite(df, "raw", paths.root, min_rows=min_rows)
    audit.event("validate", "raw_complaints", "data quality gate", rows_in=len(df), outcome="passed" if summary["success"] else "failed",
                detail={k: summary[k] for k in ("evaluated", "passed", "failed")})
    val.assert_no_critical_failures(summary)
    return summary


@task(name="clean")
def t_clean(df: pd.DataFrame, audit: AuditLog, snapshot_date: str | None) -> tuple[pd.DataFrame, dict]:
    out, rep = clean_mod.clean(df, snapshot_date)
    audit.event("clean", "complaints", "remove duplicates, fix types, apply missing-data policy", rows_in=len(df), rows_out=len(out),
                columns=out.columns, output_hash=frame_fingerprint(out), detail=rep)
    return out, rep


@task(name="transform")
def t_transform(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    out = trans.transform(df)
    audit.event("transform", "complaints", "derive labels, time keys and audit-only flags", rows_in=len(df), rows_out=len(out),
                columns=out.columns, output_hash=frame_fingerprint(out))
    return out


@task(name="integrate")
def t_integrate(df: pd.DataFrame, paths: Paths, audit: AuditLog) -> pd.DataFrame:
    ref = integ.load_state_region(paths.reference / "state_region.csv")
    out = integ.integrate(df, ref)
    audit.event("integrate", "complaints+state_region", "join region reference and company weekly volume", rows_in=len(df),
                rows_out=len(out), columns=out.columns, output_hash=frame_fingerprint(out))
    return out


@task(name="validate_curated")
def t_validate_curated(df: pd.DataFrame, paths: Paths, audit: AuditLog, min_rows: int) -> dict:
    summary = val.run_suite(df, "curated", paths.root, min_rows=min_rows)
    audit.event("validate", "curated_complaints", "data quality gate", rows_in=len(df), outcome="passed" if summary["success"] else "failed",
                detail={k: summary[k] for k in ("evaluated", "passed", "failed")})
    val.assert_no_critical_failures(summary)
    return summary


@task(name="bias_suite")
def t_bias(df: pd.DataFrame, paths: Paths, audit: AuditLog, min_block: int, min_warn: int) -> dict:
    rep = bias_mod.bias_suite(df, min_block=min_block, min_warn=min_warn)
    (paths.reports / "bias_report.json").write_text(json.dumps(rep, indent=2))
    audit.event("bias_check", "curated_complaints", "representation-bias audit (protected fields read for audit only)",
                rows_in=len(df), outcome=rep["overall"].lower(), detail={k: v["status"] for k, v in rep["checks"].items()})
    bias_mod.assert_publishable(rep)
    return rep


@task(name="dashboard_feed")
def t_feed(df: pd.DataFrame, paths: Paths, audit: AuditLog, min_total: int) -> pd.DataFrame:
    feed = integ.dashboard_feed(df, min_total=min_total)
    out = paths.processed / "dashboard_weekly_feed.parquet"
    feed.to_parquet(out, index=False)
    audit.event("publish", "dashboard_weekly_feed", "weekly company-issue counts and surge flags (goal G3)", rows_in=len(df),
                rows_out=len(feed), columns=feed.columns, output_hash=file_sha256(out))
    return feed


@task(name="anonymise")
def t_anonymise(df: pd.DataFrame, audit: AuditLog) -> tuple[pd.DataFrame, dict]:
    out, rep = anon.anonymise(df.assign(Company=df["Company"]))
    if rep["pii_remaining"]:
        raise ValueError(f"PII patterns remain after anonymisation: {rep['pii_remaining']}")
    audit.event("anonymise", "complaints", "pseudonymise ID, generalise quasi-identifiers, enforce k-anonymity", rows_in=len(df),
                rows_out=len(out), columns=out.columns, output_hash=frame_fingerprint(out), detail={"k": rep["k_anonymity"]})
    return out, rep


@task(name="publish")
def t_publish(curated: pd.DataFrame, anonymised: pd.DataFrame, paths: Paths, audit: AuditLog) -> dict:
    """Write the model-ready table and the restricted audit table (protected fields kept out of the model table)."""
    tag_cols = ["tag_recorded", "tag_older", "tag_servicemember", "tag_group"]
    base = anonymised.drop(columns=[c for c in tag_cols if c in anonymised.columns])
    flags = curated.assign(complaint_key=anon.pseudonymise(curated["Complaint ID"]))[["complaint_key", *tag_cols]]
    keyed = base.merge(flags.drop_duplicates("complaint_key"), on="complaint_key", how="left", validate="one_to_one")
    analysis, audit_tbl = integ.split_outputs(keyed)
    a_path, g_path = paths.processed / "analysis_table.parquet", paths.processed / "audit_groups.parquet"
    analysis.to_parquet(a_path, index=False)
    audit_tbl.to_parquet(g_path, index=False)
    for name, frame, p in (("analysis_table", analysis, a_path), ("audit_groups", audit_tbl, g_path)):
        audit.event("publish", name, "model-ready output" if name == "analysis_table" else "restricted fairness-audit output",
                    rows_in=len(anonymised), rows_out=len(frame), columns=frame.columns, output_hash=file_sha256(p))
    return {"analysis_table": str(a_path), "audit_groups": str(g_path), "analysis_rows": len(analysis), "analysis_columns": list(analysis.columns)}


@flow(name="cfpb-service-failure-pipeline", log_prints=True)
def pipeline(source: str, root: str | None = None, snapshot_date: str | None = None, min_rows: int = 5000,
             min_group_block: int = 100, min_group_warn: int = 1000, feed_min_total: int = 200) -> dict:
    """End-to-end ETL from the CFPB bulk file to the model-ready, anonymised analysis table."""
    log = get_run_logger()
    paths = Paths(Path(root)) if root else Paths()
    paths.make()
    audit = AuditLog(paths.audit_log)
    lin = Lineage(paths.reports / "lineage.json")
    # Governance gate: the pipeline runs as the data_engineer role and every grant is written to the audit log
    policy = load_policy(paths.policy)
    for layer, action in (("raw", "read"), ("interim", "write"), ("processed_analysis", "write"), ("processed_audit", "write"),
                          ("reports", "write"), ("audit_log", "append")):
        check_access(policy, "data_engineer", layer, action, audit)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    raw = t_ingest(source, paths, audit)
    lin.add("ingest", [str(source)], [str(paths.interim / "scope.parquet")], rows_out=len(raw),
            output_sha256=file_sha256(paths.interim / "scope.parquet"), code_version=code_hash(__import__("src.ingestion.ccdb_ingest", fromlist=["x"])))
    raw_val = t_validate_raw(raw, paths, audit, min_rows)
    lin.add("validate_raw", ["scope.parquet"], ["gx data docs: raw_complaints"], rows_in=len(raw), detail={k: raw_val[k] for k in ("evaluated", "passed", "failed")})
    cleaned, clean_rep = t_clean(raw, audit, snapshot_date)
    lin.add("clean", ["scope.parquet"], ["cleaned frame"], rows_in=len(raw), rows_out=len(cleaned), code_version=code_hash(clean_mod), detail=clean_rep)
    transformed = t_transform(cleaned, audit)
    lin.add("transform", ["cleaned frame"], ["labels, week_start, tag flags"], rows_in=len(cleaned), rows_out=len(transformed), code_version=code_hash(trans))
    curated = t_integrate(transformed, paths, audit)
    lin.add("integrate", ["transformed frame", "reference/state_region.csv"], ["curated frame"], rows_in=len(transformed),
            rows_out=len(curated), code_version=code_hash(integ))
    cur_val = t_validate_curated(curated, paths, audit, min_rows)
    lin.add("validate_curated", ["curated frame"], ["gx data docs: curated_complaints"], rows_in=len(curated), detail={k: cur_val[k] for k in ("evaluated", "passed", "failed")})
    bias_rep = t_bias(curated, paths, audit, min_group_block, min_group_warn)
    lin.add("bias_suite", ["curated frame"], ["reports/bias_report.json"], rows_in=len(curated), code_version=code_hash(bias_mod), detail={"overall": bias_rep["overall"]})
    feed = t_feed(curated, paths, audit, feed_min_total)
    lin.add("dashboard_feed", ["curated frame"], ["data/processed/dashboard_weekly_feed.parquet"], rows_in=len(curated), rows_out=len(feed),
            output_sha256=file_sha256(paths.processed / "dashboard_weekly_feed.parquet"))
    anonymised, anon_rep = t_anonymise(curated, audit)
    lin.add("anonymise", ["curated frame"], ["anonymised frame"], rows_in=len(curated), rows_out=len(anonymised), code_version=code_hash(anon),
            detail={"k_after": anon_rep["k_anonymity"]["k_after"], "pii_remaining": anon_rep["pii_remaining"]})
    published = t_publish(curated, anonymised, paths, audit)
    lin.add("publish", ["anonymised frame", "audit-only flags"], ["data/processed/analysis_table.parquet", "data/processed/audit_groups.parquet"],
            rows_in=len(anonymised), rows_out=published["analysis_rows"], output_sha256=file_sha256(paths.processed / "analysis_table.parquet"))
    lin.save()

    ok, n_events = verify_chain(paths.audit_log)
    summary = {"started": started, "finished": datetime.now(UTC).isoformat(timespec="seconds"), "source": str(source),
               "rows": {"scope": len(raw), "clean": len(cleaned), "published": published["analysis_rows"]},
               "clean_report": clean_rep, "validation": {"raw": {k: raw_val[k] for k in ("evaluated", "passed", "failed")},
                                                         "curated": {k: cur_val[k] for k in ("evaluated", "passed", "failed")}},
               "bias_overall": bias_rep["overall"], "k_anonymity": anon_rep["k_anonymity"], "pii_redactions": anon_rep["pii_redactions"],
               "outputs": published, "audit_log": {"path": str(paths.audit_log), "events": n_events, "chain_ok": ok}}
    (paths.reports / "pipeline_run.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("pipeline finished: %s", json.dumps(summary["rows"]))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/raw/complaints.csv.zip", help="CFPB bulk .zip (or a .csv extract)")
    ap.add_argument("--root", default=None, help="working root for data/, reports/, logs/ (default: repository root)")
    ap.add_argument("--snapshot-date", default=None, help="drop complaints dated after this day (YYYY-MM-DD)")
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--serve", action="store_true", help="serve the flow on a schedule instead of running once")
    ap.add_argument("--cron", default="0 6 * * *")
    a = ap.parse_args()
    if a.serve:
        pipeline.serve(name="cfpb-daily", cron=a.cron, parameters={"source": a.source, "root": a.root, "snapshot_date": a.snapshot_date})
    else:
        pipeline(a.source, a.root, a.snapshot_date, a.min_rows)
