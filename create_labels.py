import pandas as pd

# Load the GEE NDVI data
df = pd.read_csv(
    "data/Monthly_NDVI_Mysuru_Srirangapatna_KRPete_2025.csv"
)

# -----------------------------------------
# Define growth-stage labels
# -----------------------------------------

def assign_stage(ndvi):
    if ndvi < 0.45:
        return "Early Growth"
    elif ndvi < 0.60:
        return "Mid Growth"
    else:
        return "Late Growth"


# Apply labels
df["Growth_Stage"] = df["NDVI"].apply(assign_stage)


# -----------------------------------------
# Convert stages to numeric labels
# -----------------------------------------

stage_mapping = {
    "Early Growth": 0,
    "Mid Growth": 1,
    "Late Growth": 2
}

df["Stage_Label"] = df["Growth_Stage"].map(stage_mapping)


# -----------------------------------------
# Display the labelled data
# -----------------------------------------

print("\nLabelled NDVI Data:\n")

print(
    df[
        [
            "Date",
            "Location",
            "NDVI",
            "Growth_Stage",
            "Stage_Label"
        ]
    ]
)


# -----------------------------------------
# Show number of samples in each stage
# -----------------------------------------

print("\nGrowth Stage Distribution:\n")

print(
    df["Growth_Stage"].value_counts()
)


# -----------------------------------------
# Save labelled dataset
# -----------------------------------------

df.to_csv(
    "data/crop_growth_stage_dataset.csv",
    index=False
)

print(
    "\nSaved as: data/crop_growth_stage_dataset.csv"
)