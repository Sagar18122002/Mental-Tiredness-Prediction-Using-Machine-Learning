import os
import joblib

from preprocessor import preprocess_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def model_training():

    # ---------------- Load Preprocessed Data ---------------- #

    df = preprocess_data()

    # Remove prefixes
    df.columns = [col.split("__")[-1] for col in df.columns]

    # ---------------- Features & Target ---------------- #

    X = df.drop("mental_tiredness_score", axis=1)
    y = df["mental_tiredness_score"]

    # ---------------- Train Test Split ---------------- #

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- Feature Engineering ---------------- #

    X_train["workload_per_hour"] = (
        X_train["workload_score"] /
        X_train["sleep_hours"]
    )

    X_test["workload_per_hour"] = (
        X_test["workload_score"] /
        X_test["sleep_hours"]
    )

    X_train["screen_to_sleep"] = (
        X_train["screen_time_min"] /
        X_train["sleep_hours"]
    )

    X_test["screen_to_sleep"] = (
        X_test["screen_time_min"] /
        X_test["sleep_hours"]
    )

    # ---------------- Feature Selection ---------------- #

    cols_to_drop = [
        "work_environment",
        "work_type_Office",
        "work_type_Student",
        "work_type_Manual",
        "work_type_Remote"
    ]

    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    # ---------------- Hyperparameter Tuning ---------------- #

    params = {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=params,
        cv=5,
        scoring="r2"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ---------------- Save Model ---------------- #

    model_path = os.path.join(MODEL_DIR, "model.pkl")

    joblib.dump(best_model, model_path)

    print("=" * 50)
    print("Model Training Completed")
    print("=" * 50)
    print("Best Parameters :", grid.best_params_)
    print("Best CV Score   :", grid.best_score_)
    print(f"Model Saved At  : {model_path}")

    return best_model, X_test, y_test


if __name__ == "__main__":
    model_training()