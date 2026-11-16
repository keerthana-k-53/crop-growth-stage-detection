import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/Mysuru_NDVI_TimeSeries.csv")

# Remove missing values
df = df.dropna()

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# Plot
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["NDVI"], marker="o", linewidth=2)

plt.title("NDVI Time Series")
plt.xlabel("Date")
plt.ylabel("NDVI")
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/ndvi_time_series.png", dpi=300)

print("Time Series Graph Saved Successfully!")