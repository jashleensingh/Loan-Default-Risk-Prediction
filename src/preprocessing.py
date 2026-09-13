"""
Builds the preprocessing pipeline shared by training and inference, so the
exact same transformations (scaling, one-hot encoding) are guaranteed to be
applied at prediction time as were used during training — a common source
of train/serve skew when preprocessing is duplicated by hand.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "credit_score",
    "annual_income",
    "loan_amount",
    "employment_length_years",
    "debt_to_income_ratio",
    "num_previous_defaults",
    "loan_term_months",
    "interest_rate",
]

CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "default"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
