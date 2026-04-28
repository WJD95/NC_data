#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# @Time    : 2025/8/18 13:31
# @Author  : Jindi
# @email   : jindi.wang@durham.ac.uk
import pandas as pd

# Input and output file paths
in_path = "final_comments_with_predictions.xlsx"
out_path = "final_comments_means_update.xlsx"

# Load predictions
df = pd.read_excel(in_path, sheet_name="Sheet1")

# Check required columns exist
for col in ["number", "top comment", "CV", "CA"]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available columns: {list(df.columns)}")

# Group by 'number' and compute mean + count
group_means = (
    df.groupby("number")
      .agg(
          # CV_mean=("CV", "mean"),
          # CA_mean=("CA", "mean"),
          count=("CV", "size")   # or count rows in group
      )
      .reset_index()
)

df["top comment"] = df["number"].map(
    group_means.set_index("number")["count"]
)

# Save results
df.to_excel(out_path, index=False)

print(f"✅ Group mean values with counts saved to {out_path}")

