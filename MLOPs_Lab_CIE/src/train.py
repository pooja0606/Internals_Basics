import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Create folders if not exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("completion_days", axis=1)
y = df["completion_days"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("edutrack-completion-days")

results = []
best_model_obj = None

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        mlflow.set_tag("domain", "edtech")

        return model, {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }

# Train models
lasso = Lasso()
rf = RandomForestRegressor(random_state=42)

m1, r1 = evaluate(lasso, "Lasso")
m2, r2 = evaluate(rf, "RandomForest")

results.extend([r1, r2])

# Select best
best = min(results, key=lambda x: x["rmse"])
best_model_obj = m1 if best["name"] == "Lasso" else m2

# Save model
joblib.dump(best_model_obj, "models/model.pkl")

# Save JSON
output = {
    "experiment_name": "edutrack-completion-days",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 Done!")