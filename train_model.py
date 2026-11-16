import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the labelled dataset
df = pd.read_csv(
    "data/crop_growth_stage_dataset.csv"
)


# 2. Select input feature and target
X = df[["NDVI"]]
y = df["Stage_Label"]


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions on test data
y_pred = model.predict(X_test)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Model Results")
print("---------------------------")
print("Training samples:", len(X_train))
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
# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 10. Save the trained model
joblib.dump(
    model,
    "models/crop_growth_stage_model.pkl"
)

print("\nModel saved as:")
print("models/crop_growth_stage_model.pkl")