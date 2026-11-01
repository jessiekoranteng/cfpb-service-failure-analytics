"""Stage 6a: automated data validation with Great Expectations (GX 1.x).

Two suites run against the real data:
  raw_complaints      right after ingestion: schema, prohibited columns, key integrity, controlled vocabularies, row count
  curated_complaints  after clean/transform/integrate: derived labels, region mapping, ranges, base-rate sanity

Results are written to GX Data Docs (HTML) and returned as a compact summary; the flow fails closed on any failed
expectation of severity `critical`. Expectations with `mostly` < 1 are warnings by design (real data has tolerated noise).
"""
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

RAW_COLUMNS = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State", "Tags", "Submitted via",
               "Company response to consumer", "Timely response?", "Complaint ID"]
PRODUCTS = ["Checking or savings account", "Money transfer, virtual currency, or money service"]
RESPONSES = ["Closed with explanation", "Closed with monetary relief", "Closed with non-monetary relief", "In progress",
             "Untimely response"]
CHANNELS = ["Web", "Referral", "Phone", "Postal mail", "Fax", "Email"]
TAGS = ["Older American", "Servicemember", "Older American, Servicemember"]
REGIONS = ["Northeast", "Midwest", "South", "West", "Other"]
CRITICAL, WARNING = {"severity": "critical"}, {"severity": "warning"}


def raw_suite(min_rows: int = 5000) -> gx.ExpectationSuite:
    s = gx.ExpectationSuite(name="raw_complaints")
    add = s.add_expectation
    add(gxe.ExpectTableColumnsToMatchSet(column_set=RAW_COLUMNS, exact_match=True, meta={"notes": "Exactly the 12 documented columns; ZIP code, narrative and public response must never appear.", **CRITICAL}))
    add(gxe.ExpectTableRowCountToBeBetween(min_value=min_rows, max_value=5_000_000, meta={"notes": f"Module 3 requires at least {min_rows:,} records.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToNotBeNull(column="Complaint ID", meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeUnique(column="Complaint ID", meta={"notes": "Primary key.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToMatchRegex(column="Complaint ID", regex=r"^\d{5,10}$", meta=CRITICAL))
    add(gxe.ExpectColumnValuesToMatchRegex(column="Date received", regex=r"^\d{4}-\d{2}-\d{2}$", meta=CRITICAL))
    add(gxe.ExpectColumnMinToBeBetween(column="Date received", min_value="2020-01-01", max_value="2026-12-31", meta={"notes": "Scope starts 2020-01-01.", **CRITICAL}))
    for col in ("Product", "Company", "Submitted via", "Company response to consumer", "Timely response?"):
        add(gxe.ExpectColumnValuesToNotBeNull(column=col, meta=CRITICAL))
    add(gxe.ExpectColumnValuesToNotBeNull(column="Issue", mostly=0.999, meta={"notes": "Real data has 5 blank Issue values in 535,601 rows (0.001%); tolerated up to 0.1% and filled with 'Unknown' by the clean stage.", **WARNING}))
    add(gxe.ExpectColumnValuesToBeInSet(column="Product", value_set=PRODUCTS, meta={"notes": "Scope filter.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToBeInSet(column="Timely response?", value_set=["Yes", "No"], meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeInSet(column="Company response to consumer", value_set=RESPONSES, meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeInSet(column="Submitted via", value_set=CHANNELS, mostly=0.999, meta={"notes": "New channels would be tolerated up to 0.1%.", **WARNING}))
    add(gxe.ExpectColumnValuesToBeInSet(column="Tags", value_set=TAGS, meta={"notes": "Nulls are ignored; only unknown tag labels fail.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToMatchRegex(column="State", regex=r"^[A-Za-z .]{2,40}$", mostly=0.95, meta={"notes": "State may be missing (about 2.5%); present values must be letters.", **WARNING}))
    add(gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="State", min_value=0.95, max_value=1.0, meta=WARNING))
    return s


def curated_suite(min_rows: int = 5000) -> gx.ExpectationSuite:
    s = gx.ExpectationSuite(name="curated_complaints")
    add = s.add_expectation
    add(gxe.ExpectTableRowCountToBeBetween(min_value=min_rows, max_value=5_000_000, meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeUnique(column="Complaint ID", meta={"notes": "Duplicates removed by the clean stage.", **CRITICAL}))
    for col in ("Sub-product", "Issue", "Sub-issue", "State"):
        add(gxe.ExpectColumnValuesToNotBeNull(column=col, meta={"notes": "Gaps filled with 'Unknown' by the documented cleaning rule.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToBeInSet(column="untimely", value_set=[0, 1], meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeInSet(column="monetary_relief", value_set=[0.0, 1.0], meta={"notes": "Null = still in progress.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToBeInSet(column="region", value_set=REGIONS, meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeInSet(column="tag_group", value_set=["None recorded", "Older American", "Servicemember", "Both"], meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeBetween(column="year", min_value=2020, max_value=2026, meta=CRITICAL))
    add(gxe.ExpectColumnValuesToBeBetween(column="company_volume_lag4", min_value=0, meta={"notes": "Counts cannot be negative.", **CRITICAL}))
    add(gxe.ExpectColumnValuesToNotBeNull(column="week_start", meta=CRITICAL))
    add(gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="monetary_relief", min_value=0.95, max_value=1.0, meta={"notes": "In-progress share is under 1%.", **WARNING}))
    add(gxe.ExpectColumnMeanToBeBetween(column="untimely", min_value=0.002, max_value=0.05, meta={"notes": "Base-rate sanity: about 1.1% in the 2026 snapshot.", **WARNING}))
    add(gxe.ExpectColumnMeanToBeBetween(column="monetary_relief", min_value=0.03, max_value=0.25, meta={"notes": "Base-rate sanity: about 11% overall; drift is expected by year.", **WARNING}))
    add(gxe.ExpectColumnValuesToBeInSet(column="region", value_set=REGIONS[:4], mostly=0.95, meta={"notes": "At most 5% of records may map to 'Other' (territories, military, missing).", **WARNING}))
    return s


def _context(root: Path):
    return gx.get_context(mode="file", project_root_dir=str(root))


def for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """GX works on plain pandas types: ISO date strings and float labels."""
    d = df.copy()
    if "Date received" in d.columns and pd.api.types.is_datetime64_any_dtype(d["Date received"]):
        d["Date received"] = d["Date received"].dt.strftime("%Y-%m-%d")
    if "monetary_relief" in d.columns:
        d["monetary_relief"] = d["monetary_relief"].astype("float")
    return d


SUITES = {"raw": raw_suite, "curated": curated_suite}


def run_suite(df: pd.DataFrame, which: str, root, min_rows: int = 5000, build_docs: bool = True) -> dict:
    """Validate `df` against the named suite ('raw' or 'curated'); returns a summary dict (no data values)."""
    root = Path(root)
    ctx = _context(root)                      # a GX context must exist before expectations are created
    suite = SUITES[which](min_rows)
    name = suite.name
    ds = ctx.data_sources.add_or_update_pandas(name=f"{name}_source")
    asset = ds.add_dataframe_asset(name=f"{name}_asset") if f"{name}_asset" not in [a.name for a in ds.assets] else ds.get_asset(f"{name}_asset")
    bd = asset.add_batch_definition_whole_dataframe(f"{name}_batch") if f"{name}_batch" not in [b.name for b in asset.batch_definitions] else asset.get_batch_definition(f"{name}_batch")
    suite = ctx.suites.add_or_update(suite)
    vd = ctx.validation_definitions.add_or_update(gx.ValidationDefinition(name=f"{name}_validation", data=bd, suite=suite))
    cp = ctx.checkpoints.add_or_update(gx.Checkpoint(
        name=f"{name}_checkpoint", validation_definitions=[vd],
        actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")] if build_docs else [],
        result_format={"result_format": "SUMMARY"}))
    res = cp.run(batch_parameters={"dataframe": for_validation(df)})
    vr = next(iter(res.run_results.values()))
    failed = []
    for r in vr.results:
        if not r.success:
            cfg = r.expectation_config
            failed.append({"expectation": cfg.type, "column": cfg.kwargs.get("column"),
                           "severity": (cfg.meta or {}).get("severity", "critical")})
    stats = vr.statistics
    critical_failed = [f for f in failed if f["severity"] == "critical"]
    return {"suite": name, "success": bool(res.success), "evaluated": stats["evaluated_expectations"],
            "passed": stats["successful_expectations"], "failed": stats["unsuccessful_expectations"],
            "critical_failed": critical_failed, "warnings_failed": [f for f in failed if f["severity"] != "critical"],
            "data_docs": str(root / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html")}


def assert_no_critical_failures(summary: dict) -> None:
    if summary["critical_failed"]:
        raise ValueError(f"validation suite '{summary['suite']}' failed critical expectations: {summary['critical_failed']}")
