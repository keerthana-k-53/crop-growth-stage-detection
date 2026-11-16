import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load temporal feature dataset
df = pd.read_csv(
    "data/crop_growth_temporal_features.csv"
)


# 2. Select temporal input features
features = [
    "Previous_NDVI",
    "Month_Gap",
    "NDVI_Change",
    "NDVI_Trend"
]

X = df[features]
y = df["Stage_Label"]


# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# 4. Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train model
model.fit(X_train, y_train)


# 6. Predict test data
y_pred = model.predict(X_test)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTemporal Random Forest Results")
print("------------------------------")
print("Features used:")
print(features)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 8. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Early Growth",
            "Mid Growth",
            "Late Growth"
        ],
        zero_division=0
    )
)


# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 10. Feature importance
print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.to_string(index=False))


# 11. Save temporal model
joblib.dump(
    model,
    "models/crop_growth_temporal_model.pkl"
)

print("\nTemporal model saved as:")
print("models/crop_growth_temporal_model.pkl")