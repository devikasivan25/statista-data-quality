## Architecture
The pipeline operates as a pre-ingestion quality gate, using `pandas` for 
rule-based checks and `openai` for semantic validation with a heuristic fallback.

## Quality Measures Implemented
1. **Completeness:** Validates paired existence of Revenue and Currency (equivalent to SQL paired NOT NULL).
2. **Revenue Validity:** Flags non-numeric extractions and impossible financial values such as negative revenue.
3. **Year Range Validity:** Ensures temporal data falls within a logical calendar range (2000–2030).
4. **Statistical Outlier Detection:** Uses Z-score analysis per company group (similar to SQL PARTITION BY) to identify anomalous revenue values.
5. **Industry Alignment:** Uses prompt engineering to verify company-to-industry mapping logic, featuring API caching and a heuristic fallback for offline use.
