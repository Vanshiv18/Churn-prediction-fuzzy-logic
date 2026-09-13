"""
Section 8 - Data Preparation
Intelligent Customer Churn Prediction & Retention Decision Support System
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/telco_customer_churn.csv")

print("=== DIMENSIONS ===")
print(df.shape)

print("\n=== DTYPES (before fix) ===")
print(df.dtypes)

# TotalCharges is stored as object due to blank strings -> convert
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\n=== MISSING VALUES ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# These blanks correspond to tenure == 0 (brand new customers, no bill yet)
print("\nRows with missing TotalCharges - tenure values:")
print(df.loc[df["TotalCharges"].isnull(), "tenure"].unique())

# Impute with 0 (no charge accrued yet) - business-justified, not a random guess
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Duplicate check
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customerIDs:", df["customerID"].duplicated().sum())

# Drop identifier - irrelevant for modelling
df = df.drop(columns=["customerID"])

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# SeniorCitizen already 0/1 numeric; standardize other binary-like Yes/No columns
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for c in binary_cols:
    df[c] = df[c].map({"Yes": 1, "No": 0})

# Derived variable: tenure group (business-relevant bucket, useful for EDA & fuzzy tenure input)
def tenure_group(t):
    if t <= 12:
        return "0-1yr"
    elif t <= 36:
        return "1-3yr"
    else:
        return "3yr+"
df["TenureGroup"] = df["tenure"].apply(tenure_group)

# Derived variable: monthly charge band (proxy for customer value tier)
df["ChargeBand"] = pd.qcut(df["MonthlyCharges"], q=3, labels=["Low", "Medium", "High"])

# Outlier check on numeric fields (IQR method, reporting only - no removal, as
# these are legitimate high-tenure / high-bill customers, not data errors)
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col}: {n_out} potential outliers (IQR method)")

# One-hot encode remaining categorical columns for modelling
cat_cols = [c for c in df.columns
            if (df[c].dtype == "object" or str(df[c].dtype) in ("str", "string"))
            and c not in ["TenureGroup", "ChargeBand"]]
print("\nCategorical columns to encode:", cat_cols)

df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nFinal modelling dataframe shape:", df_model.shape)

df.to_csv("telco_clean.csv", index=False)
df_model.to_csv("telco_model_ready.csv", index=False)
print("\nSaved telco_clean.csv (for EDA) and telco_model_ready.csv (for modelling)")
