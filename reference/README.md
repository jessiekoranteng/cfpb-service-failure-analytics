# Reference data

`state_region.csv` maps U.S. state codes to the four U.S. Census Bureau regions (Northeast, Midwest, South, West).
Territories, military postal codes (AA, AE, AP) and freely associated states map to `Other`; any code not listed also maps to `Other`.
The pipeline integrates this table in the *integrate* stage so the raw State field never reaches the model-ready table (data minimisation, risk R-03).
