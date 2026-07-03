import os
import joblib
import pandas as pd

# ---------------- Paths ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ---------------- Load Saved Objects ---------------- #

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
scaler = joblib.load(SCALER_PATH)


def predict(data):

    # Raw Input
    raw_df = pd.DataFrame([data])

    # ---------------- Encoding ---------------- #

    transformed = preprocessor.transform(raw_df)

    transformed_df = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out()
    )

    # ---------------- Scaling ---------------- #

    numeric_cols = [
        col for col in transformed_df.columns
        if col.startswith("remainder__")
    ]

    categorical_cols = [
        col for col in transformed_df.columns
        if col not in numeric_cols
    ]

    scaled_numeric = scaler.transform(
        transformed_df[numeric_cols]
    )

    scaled_numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=numeric_cols,
        index=transformed_df.index
    )

    final_df = pd.concat(
        [
            transformed_df[categorical_cols],
            scaled_numeric_df
        ],
        axis=1
    )

    # ---------------- Rename Columns ---------------- #

    final_df.columns = [
        col.split("__")[-1]
        for col in final_df.columns
    ]

    # ---------------- Feature Engineering ---------------- #

    final_df["workload_per_hour"] = (
        final_df["workload_score"] /
        final_df["sleep_hours"]
    )

    final_df["screen_to_sleep"] = (
        final_df["screen_time_min"] /
        final_df["sleep_hours"]
    )

    # ---------------- Feature Selection ---------------- #

    cols_to_drop = [
        "work_environment",
        "work_type_Office",
        "work_type_Student",
        "work_type_Manual",
        "work_type_Remote"
    ]

    final_df = final_df.drop(columns=cols_to_drop)

    # ---------------- Arrange Columns ---------------- #

    expected_columns = [
        "mood",
        "number_of_decisions_made",
        "context_switch_count",
        "notifications_received",
        "screen_time_min",
        "deep_work_min",
        "task_complexity_avg",
        "caffeine_mg",
        "break_frequency",
        "sleep_hours",
        "deep_sleep_pct",
        "hydration_l",
        "noise_level_db",
        "temperature_c",
        "workload_score",
        "workload_per_hour",
        "screen_to_sleep"
    ]

    final_df = final_df[expected_columns]

    prediction = model.predict(final_df)

    return prediction[0]


if __name__ == "__main__":

    sample = {
        "mood": "Happy",
        "work_environment": "Quiet",
        "work_type": "Office",
        "number_of_decisions_made": 10,
        "context_switch_count": 12,
        "notifications_received": 20,
        "screen_time_min": 300,
        "deep_work_min": 180,
        "task_complexity_avg": 7,
        "caffeine_mg": 150,
        "break_frequency": 5,
        "sleep_hours": 7,
        "deep_sleep_pct": 80,
        "hydration_l": 2.5,
        "noise_level_db": 50,
        "temperature_c": 24,
        "workload_score": 75
    }

    print("Predicted Mental Tiredness Score:", predict(sample))