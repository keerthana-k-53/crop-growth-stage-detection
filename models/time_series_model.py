import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load NDVI Time Series
df = pd.read_csv("data/Mysuru_NDVI_TimeSeries.csv")

# Rename column
df = df.rename(columns={"NDVI": "ndvi"})

# Remove missing values
df = df.dropna()

# Create crop growth stages from NDVI
def growth_stage(ndvi):
    if ndvi < 0.2:
        return "Initial"
    elif ndvi < 0.4:
        return "Development"
    elif ndvi < 0.6:
        return "Mid"
    else:
        return "Late"

df["Stage"] = df["ndvi"].apply(growth_stage)

# Feature and Label
X = df[["ndvi"]]
y = df["Stage"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Model Accuracy:", round(acc * 100, 2), "%")

# Save Model
joblib.dump(model, "models/crop_growth_stage_model.pkl")

print("Model Saved Successfully!")
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Greens"
)

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()