import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Run this script from the project root:
# python train_temporal_model.py

DATA_PATH = Path("data/crop_growth_temporal_features.csv")
MODEL_PATH = Path("models/crop_growth_temporal_model.pkl")

FEATURES = [
    "Previous_NDVI",
    "Month_Gap",
    "NDVI_Change",
    "NDVI_Trend",
]
CLASS_NAMES = ["Early Growth", "Mid Growth", "Late Growth"]
CLASS_IDS = [0, 1, 2]


def calendar_month_gap(previous_date, current_date):
    """Whole calendar-month difference, matching the dashboard."""
    return (
        (current_date.year - previous_date.year) * 12
        + current_date.month
        - previous_date.month
    )


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"Location", "Date", "NDVI", "Stage_Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{DATA_PATH} is missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
    df["Stage_Label"] = pd.to_numeric(df["Stage_Label"], errors="coerce")
    df = df.dropna(subset=["Location", "Date", "NDVI", "Stage_Label"]).copy()
    df["Stage_Label"] = df["Stage_Label"].astype(int)

    # Keep the dataset's existing Previous_NDVI and NDVI_Change values.
    # The supplied CSV's Month_Gap values are fractional, while the dashboard
    # uses whole calendar months. Round the stored gap to whole months and
    # recompute the trend so training matches the dashboard's feature logic.
    df["Month_Gap"] = pd.to_numeric(df["Month_Gap"], errors="coerce").round()
    df["NDVI_Change"] = pd.to_numeric(df["NDVI_Change"], errors="coerce")
    df["Previous_NDVI"] = pd.to_numeric(df["Previous_NDVI"], errors="coerce")
    df["NDVI_Trend"] = df["NDVI_Change"] / df["Month_Gap"]
    df = df.dropna(subset=FEATURES + ["Stage_Label", "Location"]).copy()
    df = df[df["Month_Gap"] > 0].copy()

    if df["Location"].nunique() < 2:
        raise ValueError("At least two distinct locations are required for grouped evaluation.")

    X = df[FEATURES]
    y = df["Stage_Label"]
    groups = df["Location"]

    print("\nTemporal Random Forest — location-held-out evaluation")
    print("-----------------------------------------------------")
    print("Rows used:", len(df))
    print("Locations:", sorted(df["Location"].unique()))
    print("Features:", FEATURES)
    print("\nImportant: labels are NDVI-derived categories, not field-verified crop stages.")

    # Each location is held out in turn. No location appears in both the
    # training and test data for an individual fold.
    logo = LeaveOneGroupOut()
    oof_predictions = pd.Series(index=df.index, dtype="float64")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        oof_predictions.iloc[test_idx] = predictions

        held_out = groups.iloc[test_idx].unique().tolist()
        print(f"\nFold {fold} — held-out location: {held_out}")
        print("Training rows:", len(train_idx), "| Test rows:", len(test_idx))
        print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
        print(
            "Balanced accuracy:",
            round(balanced_accuracy_score(y_test, predictions), 3),
        )
        print("Macro F1:", round(
            f1_score(y_test, predictions, labels=CLASS_IDS, average="macro", zero_division=0),
            3,
        ))

    valid = oof_predictions.notna()
    y_eval = y.loc[valid].astype(int)
    pred_eval = oof_predictions.loc[valid].astype(int)

    print("\nCombined out-of-location results")
    print("--------------------------------")
    print("Test rows:", len(y_eval))
    print("Accuracy:", round(accuracy_score(y_eval, pred_eval), 3))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_eval, pred_eval), 3))
    print("Macro F1:", round(
        f1_score(y_eval, pred_eval, labels=CLASS_IDS, average="macro", zero_division=0),
        3,
    ))

    print("\nClassification report:")
    print(classification_report(
        y_eval,
        pred_eval,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted; class order 0, 1, 2):")
    print(confusion_matrix(y_eval, pred_eval, labels=CLASS_IDS))

    # Train the deployable model on all available rows after evaluation.
    final_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    final_model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print(f"\nFinal model trained on all {len(df)} usable rows and saved to:")
    print(MODEL_PATH)
    print("The reported evaluation above comes from held-out locations, not the final all-data fit.")


if __name__ == "__main__":
    main()
