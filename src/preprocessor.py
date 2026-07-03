import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mental_tiredness_score_prediction_dataset.csv.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def preprocess_data():

    # ---------------- Load Dataset ---------------- #

    df = pd.read_csv(DATA_PATH)

    # ================= Separate Target ================= #

    y = df["mental_tiredness_score"]

    X = df.drop("mental_tiredness_score", axis=1)

    # ---------------- Outlier Treatment ---------------- #

    cols = ["screen_time_min", "noise_level_db"]

    for col in cols:

        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)

        iqr = q3 - q1

        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr

        X.loc[X[col] > ub, col] = ub
        X.loc[X[col] < lb, col] = lb

    # ---------------- Encoding ---------------- #

    mood_order = [["Low", "Neutral", "Happy"]]

    env_order = [["Quiet", "Moderate Noise", "Noisy"]]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ord_mood",
                OrdinalEncoder(categories=mood_order),
                ["mood"]
            ),
            (
                "ord_env",
                OrdinalEncoder(categories=env_order),
                ["work_environment"]
            ),
            (
                "ohe",
                OneHotEncoder(sparse_output=False),
                ["work_type"]
            )
        ],
        remainder="passthrough"
    )

    transformed = preprocessor.fit_transform(X)

    transformed_df = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out()
    )

    # ---------------- Save Encoder ---------------- #

    joblib.dump(
        preprocessor,
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    # ---------------- Scaling ---------------- #

    numeric_cols = [
        col
        for col in transformed_df.columns
        if col.startswith("remainder__")
    ]

    categorical_cols = [
        col
        for col in transformed_df.columns
        if col not in numeric_cols
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        transformed_df[numeric_cols]
    )

    scaled_df = pd.DataFrame(
        scaled,
        columns=numeric_cols,
        index=transformed_df.index
    )

    # ---------------- Save Scaler ---------------- #

    joblib.dump(
        scaler,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )

    # ---------------- Final DataFrame ---------------- #

    final_scaled_df = pd.concat(
        [
            transformed_df[categorical_cols],
            scaled_df
        ],
        axis=1
    )

    # Add target back

    final_scaled_df["mental_tiredness_score"] = y.values

    print("Preprocessor saved successfully.")
    print("Scaler saved successfully.")

    return final_scaled_df