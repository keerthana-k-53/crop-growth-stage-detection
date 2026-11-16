import rasterio
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------
# Load NDVI image
# -----------------------
project_root = Path(__file__).resolve().parent.parent
tiff_path = project_root / "data" / "Mysuru_NDVI.tif"

with rasterio.open(tiff_path) as src:
    ndvi = src.read(1)

# Remove invalid pixels
mask = (ndvi >= -1) & (ndvi <= 1)
ndvi = ndvi[mask]

# Feature
X = ndvi.reshape(-1, 1)

# Labels based on NDVI
y = np.zeros(len(ndvi), dtype=int)

y[ndvi < 0] = 0
y[(ndvi >= 0) & (ndvi < 0.2)] = 1
y[(ndvi >= 0.2) & (ndvi < 0.5)] = 2
y[ndvi >= 0.5] = 3

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# Save model
model_path = project_root / "models" / "random_forest.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")