import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("production_data.csv")

# -----------------------------
# SELECT FEATURES
# -----------------------------
model_df = data[[
    "process_step",
    "machine_id",
    "time_taken_min",
    "queue_length_units",
    "average_wait_time_min",
    "temperature_C",
    "pressure_bar",
    "power_usage_kW",
    "material_type",
    "thickness_mm",
    "estimated_time_hours",
    "estimated_cost_eur",
    "risk_score_raw",
    "defect_flag"
]].copy()

numeric_cols = [
    "time_taken_min",
    "queue_length_units",
    "average_wait_time_min",
    "temperature_C",
    "pressure_bar",
    "power_usage_kW",
    "thickness_mm",
    "estimated_time_hours",
    "estimated_cost_eur",
    "risk_score_raw"
]

categorical_cols = [
    "process_step",
    "machine_id",
    "material_type"
]

for col in numeric_cols:
    model_df[col] = model_df[col].fillna(model_df[col].median())

X = model_df.drop(columns=["defect_flag"])
y = model_df["defect_flag"]

X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

numeric_after_dummies = [c for c in numeric_cols if c in X_train.columns]

pipe = Pipeline([
    ("scaler", ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_after_dummies)],
        remainder="passthrough"
    )),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, preds), 3))
print("\nClassification report:\n")
print(classification_report(y_test, preds))

model = pipe.named_steps["model"]
feature_names = numeric_after_dummies + [c for c in X_train.columns if c not in numeric_after_dummies]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)

print("\nTop positive risk drivers:")
print(coef_df.head(12))

print("\nTop negative risk drivers:")
print(coef_df.tail(12))

joblib.dump(pipe, "defect_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
coef_df.to_csv("model_coefficients.csv", index=False)

print("\nSaved:")
print("- defect_model.pkl")
print("- model_columns.pkl")
print("- model_coefficients.csv")
