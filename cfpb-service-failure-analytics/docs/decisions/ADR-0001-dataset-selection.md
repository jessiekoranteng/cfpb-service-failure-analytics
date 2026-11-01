# ADR-0001: Dataset selection

* **Status:** Accepted (Module 1, carried into Module 2)
* **Context:** The project needs real, publicly available data with at least 5,000 records that supports
  analysis of the customer impact of service failures, and that does not require confidential employer records.
* **Decision:** Use the U.S. CFPB Consumer Complaint Database, scoped to *Checking or savings account* and
  *Money transfer, virtual currency, or money service* complaints received on or after 2020-01-01.
* **Consequences:**
  * (+) Large, real, regulator-maintained, refreshed daily, free to use with attribution; 12 of 15 columns suffice.
  * (+) Contains outcome fields (company response, timeliness) that define two supervised targets.
  * (-) U.S. complaints are an analogue, not Ghanaian data; stated as a limitation (assumption A-01, risk R-06).
  * (-) Bulk file has no narrative; Tags missing on most rows; outcomes are company decisions (risks R-01, issue I-02).
* **Source:** https://www.consumerfinance.gov/data-research/consumer-complaints/
