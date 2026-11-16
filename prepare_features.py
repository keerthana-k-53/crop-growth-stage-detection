
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 1. Load original dataset (3 existing locations)
original_df = pd.read_csv(
    DATA_DIR / "crop_growth_stage_dataset.csv"
)

original_df = original_df[
    ["Location", "Date", "NDVI", "Growth_Stage"]
].copy()

# 2. Load Shravanabelagola dataset
new_df = pd.read_csv(
    DATA_DIR / "shravanabelagola_ndvi_2025.csv"
)

# Convert numeric columns safely
new_df["NDVI"] = pd.to_numeric(new_df["NDVI"], errors="coerce")

# Assign the same NDVI-threshold labels
new_df["Growth_Stage"] = pd.NA

new_df.loc[new_df["NDVI"] < 0.45, "Growth_Stage"] = "Early Growth"

new_df.loc[
    (new_df["NDVI"] >= 0.45) & (new_df["NDVI"] < 0.60),
    "Growth_Stage"
] = "Mid Growth"

new_df.loc[
    new_df["NDVI"] >= 0.60,
    "Growth_Stage"
] = "Late Growth"

# 3. Combine all four locations
df = pd.concat(
    [original_df, new_df],
    ignore_index=True
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")

# Remove missing observations
df = df.dropna(
    subset=["Date", "NDVI", "Growth_Stage"]
)

# Sort observations by location and date
df = df.sort_values(
    ["Location", "Date"]
).reset_index(drop=True)

# 4. Calculate temporal features separately for each location
grouped = df.groupby("Location", sort=False)

df["Previous_NDVI"] = grouped["NDVI"].shift(1)

df["Month_Gap"] = (
    grouped["Date"].diff().dt.days / 30.44
)

df["NDVI_Change"] = (
    df["NDVI"] - df["Previous_NDVI"]
)

df["NDVI_Trend"] = (
    df["NDVI_Change"] / df["Month_Gap"]
)

# Remove rows without valid previous observations
df = df.dropna(
    subset=[
        "Previous_NDVI",
        "Month_Gap",
        "NDVI_Change",
        "NDVI_Trend"
    ]
)

df = df[df["Month_Gap"] > 0].copy()

# 5. Add numeric labels required by the evaluation script
stage_mapping = {
    "Early Growth": 0,
    "Mid Growth": 1,
    "Late Growth": 2
}

df["Stage_Label"] = df["Growth_Stage"].map(stage_mapping)

# 6. Save the temporal dataset
output_path = DATA_DIR / "crop_growth_temporal_features.csv"

df.to_csv(output_path, index=False)

# 7. Display results
print("\nObservations by location:")
print(df.groupby("Location").size())

print("\nStage label counts:")
print(df["Stage_Label"].value_counts().sort_index())

print("\nTemporal dataset preview:")
print(
    df[
        [
            "Date",
            "Location",
            "NDVI",
            "Previous_NDVI",
            "Month_Gap",
            "NDVI_Change",
            "NDVI_Trend",
            "Growth_Stage",
            "Stage_Label"
        ]
    ].to_string(index=False)
)

print(f"\nSaved dataset to: {output_path}")
print(f"Total temporal observations: {len(df)}")

