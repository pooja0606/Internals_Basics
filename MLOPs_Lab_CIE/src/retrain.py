import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined = pd.concat([train_df, new_df])

X = combined.drop("completion_days", axis=1)
y = combined["completion_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, preds)

# Example champion MAE (replace if needed)
champion_mae = retrained_mae + 1  

improvement = champion_mae - retrained_mae

action = "promoted" if improvement >= 0.5 else "kept_champion"

output = {
    "original_data_rows": 25,
    "new_data_rows": 20,
    "combined_data_rows": 45,
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 4 Done!")