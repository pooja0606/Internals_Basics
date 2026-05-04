import pandas as pd
import numpy as np
import mlflow
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/training_data.csv")

X = df.drop("completion_days", axis=1)
y = df["completion_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

mlflow.set_experiment("edutrack-completion-days")

with mlflow.start_run(run_name="tuning-edutrack"):

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    output = {
        "search_type": "grid",
        "n_folds": 5,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-edutrack"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("✅ Task 2 Done!")