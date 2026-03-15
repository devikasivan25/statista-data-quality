import os

import pandas as pd


# ==============================================================================
# STATISTA CASE STUDY: Automated Data Quality Pipeline
# ==============================================================================

INPUT_FILE = "CaseStudy_Quality_sample25.xlsx"
OUTPUT_FILE = "CaseStudy_Quality_Checked.xlsx"

REQUIRED_COLUMNS = [
    "REVENUE",
    "unit_REVENUE",
    "timevalue",
    "providerkey",
    "companynameofficial",
    "industrycode",
]


def is_blank(value):
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def to_numeric(series):
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def validate_required_columns(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input file is missing required columns: {missing}")


def build_llm_prompt(company, industry):
    return f"""
You are a financial data auditor checking for extraction errors.

Task:
- Compare the company name with the industry label/code.
- Decide whether the industry is plausible for that company.
- Be conservative: only mark suspicious when there is a clear mismatch.

Company name: {company}
Industry: {industry}

Return exactly one word:
VALID
or
SUSPICIOUS
""".strip()


def industry_check(company, industry):
    suspicious_pairs = [
        (["cement", "steel", "mining", "textile"], "software"),
        (["bank", "capital", "finance", "insurance"], "manufacturing"),
        (["pharma", "hospital", "health"], "real estate"),
        (["shipping", "logistics", "cargo"], "financial"),
    ]
    company_lower = str(company).lower()
    industry_lower = str(industry).lower()

    for company_keywords, industry_keyword in suspicious_pairs:
        if any(keyword in company_lower for keyword in company_keywords) and industry_keyword in industry_lower:
            return "WARN"
    return "OK"


def call_openai_plausibility(company, industry):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # try:
        # from openai import OpenAI
    # except ImportError:
        # return None

    # try:
        # client = OpenAI(api_key=api_key)
        # response = client.chat.completions.create(
            # model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            # messages=[
                # {"role": "user", "content": build_llm_prompt(company, industry)}
            # ],
            # temperature=0,
            # max_tokens=3,
        # )
    # except Exception:
        # return None

    content = response.choices[0].message.content or ""
    verdict = content.strip().upper()
    if "SUSPICIOUS" in verdict:
        return "WARN"
    if "VALID" in verdict:
        return "OK"
    return None


def check_llm_plausibility(company, industry):
    llm_result = call_openai_plausibility(company, industry)
    if llm_result in {"OK", "WARN"}:
        return llm_result
    return industry_check(company, industry)


def main():
    df = pd.read_excel(INPUT_FILE)
    validate_required_columns(df)

    df["revenue_numeric"] = to_numeric(df["REVENUE"])
    df["year_numeric"] = to_numeric(df["timevalue"])

    revenue_blank = df["REVENUE"].apply(is_blank)
    currency_blank = df["unit_REVENUE"].apply(is_blank)
    year_blank = df["timevalue"].apply(is_blank)

    # Measure 1: Completeness
    df["flag_completeness"] = (revenue_blank | currency_blank).map({True: "FAIL", False: "OK"})

    # Measure 2: Revenue validity
    df["flag_revenue_validity"] = "OK"
    df.loc[~revenue_blank & df["revenue_numeric"].isna(), "flag_revenue_validity"] = "FAIL"
    df.loc[df["revenue_numeric"] < 0, "flag_revenue_validity"] = "WARN"

    # Measure 3: Year validity
    valid_year_number = df["year_numeric"].notna() & (df["year_numeric"] % 1 == 0)
    valid_year_range = df["year_numeric"].between(2000, 2030, inclusive="both")
    df["flag_invalid_year"] = "OK"
    df.loc[year_blank | ~valid_year_number | ~valid_year_range, "flag_invalid_year"] = "FAIL"

    # Measure 4: Outlier detection within providerkey groups
    provider_mean = df.groupby("providerkey")["revenue_numeric"].transform("mean")
    provider_std = df.groupby("providerkey")["revenue_numeric"].transform("std")
    provider_count = df.groupby("providerkey")["revenue_numeric"].transform("count")
    z_score = (df["revenue_numeric"] - provider_mean).abs() / provider_std

    df["flag_outlier"] = "OK"
    outlier_base = (
        df["revenue_numeric"].notna()
        & provider_std.notna()
        & (provider_std > 0)
        & (provider_count >= 3)
    )
    df.loc[outlier_base & (z_score > 2), "flag_outlier"] = "WARN"

    # Measure 5: LLM plausibility check with prompt engineering and fallback
    llm_cache = {}

    def cached_llm_check(row):
        key = (str(row["companynameofficial"]), str(row["industrycode"]))
        if key not in llm_cache:
            llm_cache[key] = check_llm_plausibility(*key)
        return llm_cache[key]

    df["flag_llm_industry"] = df.apply(cached_llm_check, axis=1)

    flag_columns = [
        "flag_completeness",
        "flag_revenue_validity",
        "flag_invalid_year",
        "flag_outlier",
        "flag_llm_industry",
    ]
    labels = [
        "1. Completeness (Revenue + Currency)",
        "2. Revenue Validity (Non-numeric / Negative)",
        "3. Invalid Year",
        "4. Outlier (Provider-level Z-Score)",
        "5. LLM Industry Mismatch",
    ]

    df["issue_count"] = (df[flag_columns] != "OK").sum(axis=1)
    df["flag_overall"] = df["issue_count"].gt(0).map({True: "CHECK", False: "OK"})

    output_df = df.drop(columns=["revenue_numeric", "year_numeric"])
    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n--- DATA QUALITY REPORT ---")
    print(f"Rows checked: {len(df)}")
    print(
        "LLM mode: OpenAI API"
        if os.getenv("OPENAI_API_KEY")
        else "LLM mode: heuristic fallback (set OPENAI_API_KEY to enable the API call)"
    )

    for column, label in zip(flag_columns, labels):
        flagged = (df[column] != "OK").sum()
        print(f"  {label}: {flagged} rows flagged")

    print(f"  Overall rows requiring review: {(df['flag_overall'] == 'CHECK').sum()}")
    print(f"\nOutput saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()