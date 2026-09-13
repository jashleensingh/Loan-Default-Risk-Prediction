"""
Loads the trained pipeline (preprocessing + model, as one joblib artifact)
and scores new applicants. Used both as a CLI for quick manual checks and
as the library import used by api.py.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run `python src/train_model.py` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_default_risk(applicant: dict) -> dict:
    """
    applicant: dict with keys credit_score, annual_income, loan_amount,
    employment_length_years, debt_to_income_ratio, num_previous_defaults,
    loan_term_months, interest_rate, home_ownership, loan_purpose.
    Returns predicted class (0/1) and the model's default probability.
    """
    pipeline = _get_pipeline()
    df = pd.DataFrame([applicant])
    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction = int(pipeline.predict(df)[0])
    return {
        "prediction": prediction,
        "default_probability": round(probability, 4),
        "risk_band": _risk_band(probability),
    }


def _risk_band(probability: float) -> str:
    if probability < 0.15:
        return "LOW"
    if probability < 0.35:
        return "MEDIUM"
    if probability < 0.60:
        return "HIGH"
    return "VERY_HIGH"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict default risk for one applicant (JSON input)")
    parser.add_argument("--json", required=True, help="Applicant fields as a JSON string")
    args = parser.parse_args()

    applicant_data = json.loads(args.json)
    result = predict_default_risk(applicant_data)
    print(json.dumps(result, indent=2))
