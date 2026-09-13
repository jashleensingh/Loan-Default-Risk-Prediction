import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import predict_default_risk, MODEL_PATH
from src.api import app

client = TestClient(app)

SAFE_APPLICANT = {
    "credit_score": 780, "annual_income": 120000, "loan_amount": 8000,
    "employment_length_years": 12, "debt_to_income_ratio": 0.08,
    "num_previous_defaults": 0, "loan_term_months": 36, "interest_rate": 6.2,
    "home_ownership": "MORTGAGE", "loan_purpose": "home_improvement",
}

RISKY_APPLICANT = {
    "credit_score": 580, "annual_income": 28000, "loan_amount": 30000,
    "employment_length_years": 0.5, "debt_to_income_ratio": 0.55,
    "num_previous_defaults": 2, "loan_term_months": 60, "interest_rate": 22.0,
    "home_ownership": "RENT", "loan_purpose": "debt_consolidation",
}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet — run src/train_model.py")
def test_risky_applicant_scores_higher_than_safe_applicant():
    safe_result = predict_default_risk(SAFE_APPLICANT)
    risky_result = predict_default_risk(RISKY_APPLICANT)

    assert 0.0 <= safe_result["default_probability"] <= 1.0
    assert 0.0 <= risky_result["default_probability"] <= 1.0
    assert risky_result["default_probability"] > safe_result["default_probability"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet — run src/train_model.py")
def test_risk_band_labels_are_consistent_with_probability():
    result = predict_default_risk(RISKY_APPLICANT)
    prob = result["default_probability"]
    band = result["risk_band"]

    if prob < 0.15:
        assert band == "LOW"
    elif prob < 0.35:
        assert band == "MEDIUM"
    elif prob < 0.60:
        assert band == "HIGH"
    else:
        assert band == "VERY_HIGH"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet — run src/train_model.py")
def test_predict_endpoint_returns_expected_shape():
    response = client.post("/predict", json=SAFE_APPLICANT)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction", "default_probability", "risk_band"}
    assert body["prediction"] in (0, 1)


def test_predict_endpoint_rejects_invalid_credit_score():
    bad_applicant = {**SAFE_APPLICANT, "credit_score": 999}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422
