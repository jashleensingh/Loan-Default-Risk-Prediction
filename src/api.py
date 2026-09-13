"""
Minimal REST API around predict.py. Run with:
    uvicorn src.api:app --reload
Then POST an applicant to http://localhost:8000/predict
"""
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_default_risk

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Scores a loan applicant's probability of default.",
    version="1.0.0",
)


class HomeOwnership(str, Enum):
    RENT = "RENT"
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"


class LoanPurpose(str, Enum):
    debt_consolidation = "debt_consolidation"
    credit_card = "credit_card"
    home_improvement = "home_improvement"
    major_purchase = "major_purchase"
    other = "other"


class Applicant(BaseModel):
    credit_score: int = Field(ge=300, le=850)
    annual_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    employment_length_years: float = Field(ge=0, le=50)
    debt_to_income_ratio: float = Field(ge=0, le=1)
    num_previous_defaults: int = Field(ge=0)
    loan_term_months: int
    interest_rate: float = Field(ge=0)
    home_ownership: HomeOwnership
    loan_purpose: LoanPurpose


class PredictionResponse(BaseModel):
    prediction: int
    default_probability: float
    risk_band: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant):
    return predict_default_risk(applicant.model_dump())
