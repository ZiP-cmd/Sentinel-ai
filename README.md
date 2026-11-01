# Sentinel AI

**Fraud Detection, Investigation Prioritization, and Evidence-Grounded AI Safety**

Sentinel AI is a Python-based machine learning portfolio project that identifies potentially fraudulent credit card transactions, evaluates investigation workloads, and demonstrates techniques for validating AI-generated investigation claims.

The system combines fraud classification, a FastAPI prediction service, an interactive Streamlit dashboard, structured evidence validation, safe investigation summaries, and audit-access controls.

## Features

- Credit card fraud detection using class-weighted logistic regression.
- Time-ordered training, validation, and test splits.
- Precision, recall, average precision, and confusion-matrix evaluation.
- Top-k investigation budgets and hypothetical cost analysis.
- FastAPI prediction and health endpoints.
- Streamlit dashboard for model analysis and AI safety demonstrations.
- Structured claim validation and rejection of unsupported fraud accusations.
- Sanitized investigator summaries and separate audit records.
- Demonstration role-based audit authorization, permission revocation, and simulated authorization outages.
- 81 passing automated tests at the latest recorded checkpoint.

## Model Evaluation

The final model is a scikit-learn pipeline using StandardScaler and class-weighted LogisticRegression.

The final evaluation used a previously held-out, time-ordered test split.

| Metric | Held-out test result |
|---|---:|
| Test transactions | 56,962 |
| Actual fraudulent transactions | 75 |
| Average precision | 0.7436 |
| Investigation budget | Top 100 scored transactions |
| True positives | 59 |
| False positives | 41 |
| False negatives | 16 |
| Precision at 100 | 0.5900 |
| Recall at 100 | 0.7867 |

The model's numerical output should not be interpreted as a calibrated probability that fraud occurred. A model alert identifies a transaction for review; it does not prove fraud.

These results describe retrospective evaluation on a historical dataset, not performance in a live financial institution.

## AI Investigation Safety

Sentinel demonstrates a controlled evidence-grounding workflow:

1. Validate transaction data and model decisions.
2. Accept structured candidate investigation claims.
3. Compare those claims against trusted evidence.
4. Reject unsupported claims, such as fabricated locations or accusations of confirmed fraud.
5. Reconstruct an investigator-facing summary from verified information.
6. Create a separate audit record containing validation outcomes and rejected claims.

The current implementation uses mock LLM behavior and controlled summaries. It does not integrate a live generative AI model or provide general-purpose factual verification of arbitrary natural-language responses.

## Security

The project includes local demonstrations of role-based audit authorization, session-based access checks, permission revocation, and fail-closed behavior during simulated authorization outages.

These are educational security controls rather than a production authentication service. Demo users are not securely authenticated, and audit records are not yet stored in a protected persistent audit system.

## Technology

Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, Streamlit, matplotlib, and pytest.

## Setup

The project was developed in a Windows Python 3.15 environment.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest tests/ -v
```

The large source dataset and serialized model files are excluded from Git. A fresh checkout will require the dataset and a reproducible model-training process before model-dependent functionality can run.

## Project Status

Sentinel AI is a portfolio and educational system. It has not been validated for production financial decisions, and it should not be used to make autonomous fraud accusations or account-suspension decisions.