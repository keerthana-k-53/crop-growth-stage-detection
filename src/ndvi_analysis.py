import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ----------------------------
# Locate the TIFF file
# ----------------------------
project_root = Path(__file__).resolve().parent.parent
tiff_path = project_root / "data" / "Mysuru_NDVI.tif"

# Read NDVI image
with rasterio.open(tiff_path) as src:
    ndvi = src.read(1)

# Remove invalid values
ndvi = np.where(ndvi < -1, np.nan, ndvi)

# Classification
water = np.sum(ndvi < 0)
bare = np.sum((ndvi >= 0) & (ndvi < 0.2))
moderate = np.sum((ndvi >= 0.2) & (ndvi < 0.5))
healthy = np.sum(ndvi >= 0.5)

total = water + bare + moderate + healthy

results = pd.DataFrame({
    "Category": [
        "Water",
        "Bare Soil",
        "Moderate Vegetation",
        "Healthy Vegetation"
    ],
    "Pixels": [
        water,
        bare,
        moderate,
        healthy
    ]
})

results["Percentage"] = (results["Pixels"] / total * 100).round(2)

print(results)



# Save CSV
output_csv = project_root / "outputs" / "ndvi_statistics.csv"
results.to_csv(output_csv, index=False)

# Create Bar Chart
plt.figure(figsize=(8,5))
plt.bar(results["Category"], results["Pixels"])
plt.title("NDVI Vegetation Classification")
plt.ylabel("Pixel Count")
plt.xticks(rotation=10)

chart_path = project_root / "outputs" / "ndvi_bar_chart.png"
plt.savefig(chart_path)
plt.show()

print("\nFiles Saved:")
print(output_csv)
print(chart_path)