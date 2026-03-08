from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# =========================================================
# Configuration
# =========================================================
CSV_PATH = "dummy_climate_fertility_h1.csv"   # 현재 폴더에 csv가 있다고 가정
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

print("\n[1] Data loaded")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# 2. Basic checks
# =========================================================
required_cols = [
    "state", "year", "month",
    "hot_days_gt80f",
    "precip_share_001_050", "precip_share_gt050",
    "delta_birth_8m_pct", "delta_birth_9m_pct", "delta_birth_10m_pct"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("\n[2] Missing values summary")
print(df[required_cols].isna().sum())

# Drop missing rows only for required variables
df = df.dropna(subset=required_cols).copy()

# Ensure types
df["state"] = df["state"].astype(str)
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

# =========================================================
# 3. Descriptive statistics
# =========================================================
desc = df[[
    "hot_days_gt80f",
    "delta_birth_8m_pct",
    "delta_birth_9m_pct",
    "delta_birth_10m_pct",
    "precip_share_001_050",
    "precip_share_gt050"
]].describe()

desc_path = OUTPUT_DIR / "descriptive_statistics.csv"
desc.to_csv(desc_path, encoding="utf-8-sig")

print("\n[3] Descriptive statistics")
print(desc)

# =========================================================
# 4. Correlation check
# =========================================================
corr = df[[
    "hot_days_gt80f",
    "delta_birth_8m_pct",
    "delta_birth_9m_pct",
    "delta_birth_10m_pct"
]].corr()

corr_path = OUTPUT_DIR / "correlation_matrix.csv"
corr.to_csv(corr_path, encoding="utf-8-sig")

print("\n[4] Correlation matrix")
print(corr)

# =========================================================
# 5. Hypothesis 1 regression models
#    Main structure:
#    outcome ~ hot_days_gt80f + precipitation controls + state FE + month FE + year FE
# =========================================================
formula_8 = """
delta_birth_8m_pct ~ hot_days_gt80f
+ precip_share_001_050
+ precip_share_gt050
+ C(state)
+ C(month)
+ C(year)
"""

formula_9 = """
delta_birth_9m_pct ~ hot_days_gt80f
+ precip_share_001_050
+ precip_share_gt050
+ C(state)
+ C(month)
+ C(year)
"""

formula_10 = """
delta_birth_10m_pct ~ hot_days_gt80f
+ precip_share_001_050
+ precip_share_gt050
+ C(state)
+ C(month)
+ C(year)
"""

# state-level clustered SE
model_8 = smf.ols(formula=formula_8, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["state"]}
)

model_9 = smf.ols(formula=formula_9, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["state"]}
)

model_10 = smf.ols(formula=formula_10, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["state"]}
)

# =========================================================
# 6. Extract compact result table
# =========================================================
def extract_result(model, outcome_name):
    coef = model.params.get("hot_days_gt80f", np.nan)
    se = model.bse.get("hot_days_gt80f", np.nan)
    tval = model.tvalues.get("hot_days_gt80f", np.nan)
    pval = model.pvalues.get("hot_days_gt80f", np.nan)
    ci = model.conf_int().loc["hot_days_gt80f"] if "hot_days_gt80f" in model.params.index else [np.nan, np.nan]

    return {
        "outcome": outcome_name,
        "coef_hot_days_gt80f": coef,
        "std_err": se,
        "t_value": tval,
        "p_value": pval,
        "ci_lower_95": ci[0],
        "ci_upper_95": ci[1],
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared
    }

results = pd.DataFrame([
    extract_result(model_8, "delta_birth_8m_pct"),
    extract_result(model_9, "delta_birth_9m_pct"),
    extract_result(model_10, "delta_birth_10m_pct")
])

results_path = OUTPUT_DIR / "hypothesis1_regression_results.csv"
results.to_csv(results_path, index=False, encoding="utf-8-sig")

print("\n[5] Regression summary table")
print(results)

# =========================================================
# 7. Save full text summaries
# =========================================================
with open(OUTPUT_DIR / "model_8_summary.txt", "w", encoding="utf-8") as f:
    f.write(model_8.summary().as_text())

with open(OUTPUT_DIR / "model_9_summary.txt", "w", encoding="utf-8") as f:
    f.write(model_9.summary().as_text())

with open(OUTPUT_DIR / "model_10_summary.txt", "w", encoding="utf-8") as f:
    f.write(model_10.summary().as_text())

# =========================================================
# 8. Simple interpretation logic
# =========================================================
def interpret_row(row):
    coef = row["coef_hot_days_gt80f"]
    p = row["p_value"]

    direction = "negative" if coef < 0 else "positive"
    significance = "significant" if p < 0.05 else "not significant"

    return f"{row['outcome']}: coef={coef:.4f}, p={p:.4f}, {direction}, {significance}"

print("\n[6] Interpretation")
for _, row in results.iterrows():
    print(interpret_row(row))

# =========================================================
# 9. Visualization
# =========================================================
# 9-1 Scatter: hot days vs delta_birth_9m_pct
plt.figure(figsize=(8, 5))
plt.scatter(df["hot_days_gt80f"], df["delta_birth_9m_pct"], alpha=0.7)
plt.xlabel("Hot days > 80°F")
plt.ylabel("Birth change after 9 months (%)")
plt.title("Hot Days and Birth Change at t+9")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_hotdays_vs_birth9.png", dpi=200)
plt.close()

# 9-2 Coefficient comparison plot
plot_df = results.copy()

plt.figure(figsize=(8, 5))
plt.errorbar(
    x=plot_df["outcome"],
    y=plot_df["coef_hot_days_gt80f"],
    yerr=1.96 * plot_df["std_err"],
    fmt="o",
    capsize=6
)
plt.axhline(0, linestyle="--")
plt.ylabel("Coefficient on hot_days_gt80f")
plt.title("Hypothesis 1: Effect of Hot Days on Future Birth Changes")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coef_comparison_h1.png", dpi=200)
plt.close()

# =========================================================
# 10. Optional: simpler model without FE for comparison
# =========================================================
simple_model_9 = smf.ols(
    "delta_birth_9m_pct ~ hot_days_gt80f",
    data=df
).fit()

with open(OUTPUT_DIR / "simple_model_9_summary.txt", "w", encoding="utf-8") as f:
    f.write(simple_model_9.summary().as_text())

print("\n[7] Files saved in:", OUTPUT_DIR.resolve())
print("- descriptive_statistics.csv")
print("- correlation_matrix.csv")
print("- hypothesis1_regression_results.csv")
print("- model_8_summary.txt")
print("- model_9_summary.txt")
print("- model_10_summary.txt")
print("- simple_model_9_summary.txt")
print("- scatter_hotdays_vs_birth9.png")
print("- coef_comparison_h1.png")

print("\nDone.")