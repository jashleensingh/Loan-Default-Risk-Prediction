# Loan Default Risk Prediction

A binary classifier that predicts whether a loan applicant will default,
given standard underwriting features (credit score, income, DTI ratio,
prior defaults, etc.), served over a small REST API.

## Data

No public dataset is downloaded — this ships a synthetic-data generator
(`data/generate_synthetic_data.py`) instead, so the whole project runs
offline with no dataset licensing question attached. The generator doesn't
just assign random labels: default risk is built from a logistic combination
of realistic risk factors (lower credit score, higher debt-to-income ratio,
more prior defaults, higher interest rate all push risk up) plus Gaussian
noise, calibrated to land on an ~18% overall default rate — close to
real subprime-portfolio default rates, and importantly *not* perfectly
separable, so the classification problem behaves like a real one.

| Feature | Description |
|---|---|
| `credit_score` | 300–850 |
| `annual_income` | log-normal distributed |
| `loan_amount` | requested amount |
| `employment_length_years` | years at current job |
| `debt_to_income_ratio` | 0–1 |
| `num_previous_defaults` | count |
| `loan_term_months` | 36 or 60 |
| `interest_rate` | correlated with credit score + noise |
| `home_ownership` | RENT / MORTGAGE / OWN |
| `loan_purpose` | debt_consolidation / credit_card / home_improvement / major_purchase / other |

## Approach

Two models are trained and compared rather than picking one blind:
**Logistic Regression** (interpretable baseline, coefficients are directly
readable) and **Random Forest** (captures nonlinear interactions). Both use
`class_weight="balanced"` since defaults are the minority class (~18%) —
without it, a model can get 82% accuracy by just predicting "no default"
for everyone, which is useless for actual risk screening. The model with
the higher ROC-AUC on a held-out 20% test set is persisted; the metrics
report below is real output from this repo's own `src/train_model.py` run,
not invented numbers.

### Results (this run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (selected) | 0.793 | 0.462 | 0.859 | 0.601 | **0.888** |
| Random Forest | 0.782 | 0.446 | 0.838 | 0.582 | 0.882 |

Logistic Regression edged out on ROC-AUC and was selected. Recall (86%) is
prioritized over precision by design — for loan-default screening, missing
an actual default (false negative) is more costly than flagging a safe
applicant for a second look (false positive), so `class_weight="balanced"`
was chosen deliberately over optimizing for raw accuracy. Full numbers
(including the confusion matrix) are written to `models/metrics.json` on
every training run, not just quoted here.

## Project layout

```
data/generate_synthetic_data.py   builds data/loan_data.csv
src/
  preprocessing.py    shared ColumnTransformer (scaling + one-hot), used
                       identically at train and inference time
  train_model.py       trains + compares both models, saves the best one
  predict.py            loads the saved pipeline, scores one applicant (CLI + library)
  api.py                 FastAPI wrapper: POST /predict
models/                model.joblib, metrics.json, (feature_importance.csv if RF wins)
tests/test_model.py    risk-ordering, API contract, and validation tests
```

## Running it

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data/generate_synthetic_data.py   # writes data/loan_data.csv
python src/train_model.py                # trains, evaluates, saves models/model.joblib

# CLI prediction
python src/predict.py --json '{"credit_score": 610, "annual_income": 38000, "loan_amount": 22000, "employment_length_years": 1.2, "debt_to_income_ratio": 0.42, "num_previous_defaults": 1, "loan_term_months": 60, "interest_rate": 18.5, "home_ownership": "RENT", "loan_purpose": "debt_consolidation"}'

# REST API
uvicorn src.api:app --reload
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...same fields...}'
```

`pytest` runs the full test suite (model-behavior tests skip automatically
if you haven't trained a model yet).

## Possible extensions

- Swap the synthetic generator for a real dataset (e.g. LendingClub) once
  one is available — nothing else in the pipeline needs to change, since
  `preprocessing.py` and the training/serving code are dataset-agnostic
  as long as the column names match.
- Add SHAP values to `/predict` for per-applicant explainability, not just
  a probability.
- Threshold tuning: the default 0.5 decision threshold isn't necessarily
  optimal for a screening use case — a precision-recall tradeoff analysis
  would inform where to actually cut LOW/MEDIUM/HIGH/VERY_HIGH bands.
