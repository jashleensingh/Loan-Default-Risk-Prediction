"""
Generates a synthetic loan-applicant dataset with a realistic, noisy
relationship between applicant features and default risk.

A public labeled loan-default dataset would normally be used here, but this
project ships its own generator so it runs completely offline with no
download step. The underlying risk function is a logistic combination of
weighted features plus Gaussian noise, so the resulting problem is
learnable but not perfectly separable — like real credit risk data.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 8000


def generate(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    credit_score = rng.integers(300, 851, n_samples)
    annual_income = np.round(rng.lognormal(mean=10.8, sigma=0.45, size=n_samples), 2)
    loan_amount = np.round(rng.uniform(1000, 45000, n_samples), 2)
    employment_length_years = np.round(rng.exponential(scale=5, size=n_samples), 1)
    employment_length_years = np.clip(employment_length_years, 0, 40)
    debt_to_income_ratio = np.round(rng.beta(a=2, b=5, size=n_samples), 3)  # skewed toward lower DTI
    num_previous_defaults = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    loan_term_months = rng.choice([36, 60], size=n_samples, p=[0.6, 0.4])
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.45, 0.40, 0.15])
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"],
        size=n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    # Higher credit score -> lower rate; small random lender-specific noise added
    interest_rate = np.round(22 - (credit_score - 300) / 550 * 16 + rng.normal(0, 1.2, n_samples), 2)
    interest_rate = np.clip(interest_rate, 4.0, 28.0)

    # --- Underlying (noisy, non-deterministic) default risk function ---
    z = (
        -0.014 * (credit_score - 650)
        + 2.6 * debt_to_income_ratio
        + 0.00003 * loan_amount
        - 0.00002 * annual_income
        - 0.05 * employment_length_years
        + 0.9 * num_previous_defaults
        + 0.10 * interest_rate
        + (loan_term_months == 60) * 0.25
        + rng.normal(0, 1.4, n_samples)  # noise: keeps this from being perfectly separable
    )
    default_probability = 1 / (1 + np.exp(-z + 6.5))  # shifted so overall default rate lands ~15-20%
    default = rng.binomial(1, default_probability)

    return pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "employment_length_years": employment_length_years,
        "debt_to_income_ratio": debt_to_income_ratio,
        "num_previous_defaults": num_previous_defaults,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/loan_data.csv", index=False)
    print(f"Wrote {len(df)} rows to data/loan_data.csv")
    print(f"Default rate: {df['default'].mean():.3f}")
