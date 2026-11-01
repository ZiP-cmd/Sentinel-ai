from cost_analysis import estimate_cost
from pathlib import Path
from investigation_evidence import build_investigation_evidence
from safe_response_builder import build_safe_response
import pandas as pd
import streamlit as st
from alert_budget import evaluate_alert_budget
from model_evaluation import evaluate_predictions
from budget_optimizer import compare_budget_costs
from prediction_service import load_model, predict_transaction

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"

st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_validation_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(
        "Time", kind="stable"
    ).reset_index(drop=True)

    train_end = int(len(df) * 0.60)
    validation_end = int(len(df) * 0.80)

    return df.iloc[train_end:validation_end].copy()


st.title("🛡️ Sentinel AI")
st.subheader("Fraud Investigation Dashboard")

st.caption(
    "Portfolio demonstration using historical validation data. "
    "Not a production fraud decision system."
)

artifact = get_model()
validation = get_validation_data()

st.sidebar.header("Investigation Settings")

position = st.sidebar.number_input(
    "Validation transaction number",
    min_value=1,
    max_value=len(validation),
    value=1,
    step=1,
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
)

row = validation.iloc[int(position) - 1]

# The correct fraud label is deliberately excluded
# from the information sent to the model.
transaction = row.drop("Class").to_dict()

result = predict_transaction(
    transaction,
    artifact,
    threshold=threshold,
)

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Transaction Amount",
    f"${transaction['Amount']:,.2f}",
)

col2.metric(
    "Fraud Score",
    f"{result['fraud_score']:.4f}",
)

col3.metric(
    "Decision Threshold",
    f"{result['threshold']:.2f}",
)

st.subheader("Prediction Result")

if result["flagged"]:
    st.warning("⚠️ FLAGGED FOR REVIEW")
else:
    st.success("NOT FLAGGED")

st.progress(result["fraud_score"])

st.caption(
    "The fraud score is not necessarily a calibrated "
    "probability. Changing the threshold changes the "
    "flagging decision, not the model's score."
)

with st.expander("View transaction features"):
    st.dataframe(
        pd.DataFrame(
            list(transaction.items()),
            columns=["Feature", "Value"],
        ),
        hide_index=True,
    )

if st.checkbox("Reveal historical fraud label"):
    actual_label = int(row["Class"])

    if actual_label == 1:
        st.write("Actual label: Fraud")
    else:
        st.write("Actual label: Legitimate")

    predicted_label = int(result["flagged"])

    if predicted_label == 1 and actual_label == 1:
        st.info("Outcome: True Positive")
    elif predicted_label == 1 and actual_label == 0:
        st.info("Outcome: False Positive")
    elif predicted_label == 0 and actual_label == 1:
        st.info("Outcome: False Negative")
    else:
        st.info("Outcome: True Negative")

st.divider()

st.subheader("Threshold Comparison")

st.write(
    "See how Sentinel's decision changes "
    "at different fraud detection thresholds."
)

thresholds = [0.20, 0.50, 0.70]

comparison = []

for test_threshold in thresholds:
    prediction = predict_transaction(
        transaction,
        artifact,
        threshold=test_threshold,
    )

    comparison.append({
        "Threshold": test_threshold,
        "Fraud Score": prediction["fraud_score"],
        "Decision": (
            "FLAGGED"
            if prediction["flagged"]
            else "NOT FLAGGED"
        ),
    })

comparison_df = pd.DataFrame(comparison)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

import numpy as np
from sklearn.metrics import precision_score, recall_score

st.divider()
st.header("Batch Model Evaluation")

st.write(
    "Evaluate Sentinel's performance across the "
    "entire historical validation dataset."
)

# Separate the actual fraud labels from model inputs.
X_validation = validation[artifact["features"]]
y_validation = validation["Class"].to_numpy()

# Generate model scores for every validation transaction.
model = artifact["model"]
scores = model.predict_proba(X_validation)[:, 1]

# Apply the threshold selected in the dashboard sidebar.
predictions = (scores >= threshold).astype(int)


# Evaluate the model using our tested function.
metrics = evaluate_predictions(
    y_true=y_validation,
    scores=scores,
    threshold=threshold,
)

tp = metrics["tp"]
tn = metrics["tn"]
fp = metrics["fp"]
fn = metrics["fn"]

precision = metrics["precision"]
recall = metrics["recall"]
total_alerts = metrics["alerts"]

# Calculate model performance.
precision = precision_score(
    y_validation, predictions, zero_division=0
)

recall = recall_score(
    y_validation, predictions, zero_division=0
)

total_alerts = tp + fp

st.subheader("Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Precision", f"{precision:.2%}")

with col2:
    st.metric("Recall", f"{recall:.2%}")

with col3:
    st.metric("Fraud Alerts", f"{total_alerts:,}")

st.subheader("Confusion Matrix Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("True Positives", tp)
col2.metric("False Positives", fp)
col3.metric("False Negatives", fn)
col4.metric("True Negatives", tn)

matrix = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp],
    ],
    index=["Actually Legitimate", "Actually Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"],
)

st.dataframe(matrix, use_container_width=True)

st.caption(
    f"Evaluated {len(validation):,} validation transactions "
    f"using a threshold of {threshold:.2f}."
)

if total_alerts == 0:
    st.info(
        "No fraud alerts were generated. "
        "Precision is displayed as 0 by convention."
    )

st.divider()

st.header("Investigation Budget Analysis")

st.write(
    "Evaluate how many fraud cases Sentinel can detect "
    "when investigators have limited review capacity."
)

budget = st.select_slider(
    "Maximum number of investigations",
    options=[30, 60, 100, 200],
    value=60,
)

budget_results = evaluate_alert_budget(
    y_true=y_validation,
    scores=scores,
    budget=budget,
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Fraud Detected",
    budget_results["tp"],
)

col2.metric(
    "False Alarms",
    budget_results["fp"],
)

col3.metric(
    "Precision",
    f"{budget_results['precision']:.2%}",
)

col4.metric(
    "Recall",
    f"{budget_results['recall']:.2%}",
)

st.caption(
    "Historical validation results. "
    "This budget applies to the entire validation set, "
    "not to a daily investigation limit."
)

st.divider()

st.subheader("Investigation Budget Comparison")

budgets = [30, 60, 100, 200]
comparison_rows = []

for review_budget in budgets:
    result = evaluate_alert_budget(
        y_true=y_validation,
        scores=scores,
        budget=review_budget,
    )

    comparison_rows.append({
        "Investigations": review_budget,
        "Fraud Caught": result["tp"],
        "Fraud Missed": result["fn"],
        "False Alarms": result["fp"],
        "Precision": result["precision"],
        "Recall": result["recall"],
    })

budget_df = pd.DataFrame(comparison_rows)

st.dataframe(
    budget_df.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Cost of Expanding Investigations")

budget_60 = evaluate_alert_budget(
    y_true=y_validation,
    scores=scores,
    budget=60,
)

budget_100 = evaluate_alert_budget(
    y_true=y_validation,
    scores=scores,
    budget=100,
)

extra_investigations = 100 - 60

extra_fraud = (
    budget_100["tp"] - budget_60["tp"]
)

extra_false_alarms = (
    budget_100["fp"] - budget_60["fp"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Extra Investigations",
    extra_investigations,
)

col2.metric(
    "Additional Fraud Found",
    extra_fraud,
)

col3.metric(
    "Additional False Alarms",
    extra_false_alarms,
)

if extra_fraud > 0:
    investigations_per_fraud = (
        extra_investigations / extra_fraud
    )

    st.info(
        "Expanding from 60 to 100 investigations "
        f"requires {investigations_per_fraud:.1f} "
        "additional reviews per additional "
        "fraud case detected."
    )
else:
    st.warning(
        "Expanding the investigation budget "
        "did not detect any additional fraud."
    )

st.divider()

st.header("Investigation Budget Comparison")

budgets = [30, 60, 100, 200]
comparison_rows = []

for current_budget in budgets:
    result = evaluate_alert_budget(
        y_true=y_validation,
        scores=scores,
        budget=current_budget,
    )

    comparison_rows.append({
        "Investigation Budget": current_budget,
        "Fraud Caught": result["tp"],
        "False Alarms": result["fp"],
        "Fraud Missed": result["fn"],
        "Precision": result["precision"],
        "Recall": result["recall"],
    })

budget_comparison = pd.DataFrame(comparison_rows)

# Calculate how outcomes change between budget levels.
budget_comparison["Additional Fraud Caught"] = (
    budget_comparison["Fraud Caught"]
    .diff()
    .fillna(0)
    .astype(int)
)

budget_comparison["Additional False Alarms"] = (
    budget_comparison["False Alarms"]
    .diff()
    .fillna(0)
    .astype(int)
)

st.dataframe(
    budget_comparison.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Fraud Detection by Investigation Budget")

chart_data = budget_comparison.set_index(
    "Investigation Budget"
)[["Fraud Caught", "Fraud Missed"]]

st.bar_chart(chart_data)

st.caption(
    "Historical validation results only. "
    "Additional counts compare each budget with "
    "the preceding budget in the table."
)


st.divider()

st.header("Investigation Budget Comparison")

st.write(
    "Compare how investigation capacity affects "
    "fraud detection, false alarms, precision, and recall."
)

budgets = [30, 60, 100, 200]
comparison_rows = []

for current_budget in budgets:
    result = evaluate_alert_budget(
        y_true=y_validation,
        scores=scores,
        budget=current_budget,
    )

    comparison_rows.append({
        "Budget": current_budget,
        "Fraud Caught": result["tp"],
        "False Alarms": result["fp"],
        "Fraud Missed": result["fn"],
        "Precision": result["precision"],
        "Recall": result["recall"],
    })

budget_df = pd.DataFrame(comparison_rows)

# Calculate the additional fraud detected as
# investigation capacity increases.
budget_df["Additional Fraud Caught"] = (
    budget_df["Fraud Caught"].diff()
)

budget_df["Additional Investigations"] = (
    budget_df["Budget"].diff()
)

# Measure how many additional investigations
# are needed for each additional fraud case detected.
budget_df["Reviews per Extra Fraud"] = (
    budget_df["Additional Investigations"]
    / budget_df["Additional Fraud Caught"].replace(0, float("nan"))
)

# Measure additional fraud detected per 10 reviews.
budget_df["Extra Fraud per 10 Reviews"] = (
    budget_df["Additional Fraud Caught"]
    / budget_df["Additional Investigations"]
    * 10
)

st.dataframe(
    budget_df.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "Additional Fraud Caught": "{:.0f}",
        "Additional Investigations": "{:.0f}",
        "Reviews per Extra Fraud": "{:.2f}",
        "Extra Fraud per 10 Reviews": "{:.2f}",
    }, na_rep="—"),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "These are historical validation results. "
    "The budget is applied across the entire validation dataset."
)

st.divider()
st.header("Financial Impact Analysis")

st.caption(
    "Hypothetical scenario analysis, not actual bank losses "
    "or validated monetary savings."
)

cost_per_review = st.number_input(
    "Cost per investigation ($)",
    min_value=0.0,
    value=5.0,
    step=1.0,
)

loss_per_missed_fraud = st.number_input(
    "Estimated preventable loss per missed fraud ($)",
    min_value=0.0,
    value=200.0,
    step=50.0,
)

financial_rows = []

for current_budget in [30, 60, 100, 200]:
    result = evaluate_alert_budget(
        y_true=y_validation,
        scores=scores,
        budget=current_budget,
    )

    costs = estimate_cost(
        alerts=current_budget,
        missed_fraud=result["fn"],
        cost_per_review=cost_per_review,
        loss_per_missed_fraud=loss_per_missed_fraud,
    )

    financial_rows.append({
        "Budget": current_budget,
        "Fraud Caught": result["tp"],
        "Fraud Missed": result["fn"],
        "Review Cost": costs["review_cost"],
        "Missed Fraud Cost": costs["missed_fraud_cost"],
        "Total Cost": costs["total_cost"],
    })

financial_df = pd.DataFrame(financial_rows)

st.dataframe(
    financial_df.style.format({
        "Review Cost": "${:,.2f}",
        "Missed Fraud Cost": "${:,.2f}",
        "Total Cost": "${:,.2f}",
    }),
    hide_index=True,
    use_container_width=True,
)

best_row = financial_df.loc[
    financial_df["Total Cost"].idxmin()
]

st.metric(
    "Lowest Modeled Cost — Budget",
    int(best_row["Budget"]),
)

st.metric(
    "Lowest Modeled Total Cost",
    f"${best_row['Total Cost']:,.2f}",
)

st.caption(
    "Lowest cost among the four tested budgets only. "
    "Results depend on the entered assumptions."
)

st.divider()
st.header("Automated Investigation Budget Recommendation")

st.write(
    "Compare investigation budgets using the financial "
    "assumptions selected above."
)

optimization = compare_budget_costs(
    y_true=y_validation,
    scores=scores,
    budgets=[30, 60, 100, 200],
    cost_per_review=cost_per_review,
    loss_per_missed_fraud=loss_per_missed_fraud,
)

best_budget = optimization["recommended_budget"]
lowest_cost = optimization["lowest_total_cost"]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Recommended Investigation Budget",
        best_budget,
    )

with col2:
    st.metric(
        "Lowest Modeled Cost",
        f"${lowest_cost:,.2f}",
    )

st.info(
    f"Under the selected financial assumptions, "
    f"investigating the top {best_budget} transactions "
    f"produces the lowest modeled cost among the "
    f"four budgets evaluated."
)

st.caption(
    "This is a hypothetical recommendation based on "
    "historical validation data. It is not an "
    "approved real-world fraud investigation policy."
)

st.divider()

st.header("AI Investigation Safety Demonstration")

st.write(
    "Test how Sentinel handles verified claims "
    "and unsupported AI-generated statements."
)

scenario = st.selectbox(
    "Choose a simulated AI response",
    [
        "Verified claims",
        "Unsupported fraud accusation",
        "Fabricated location",
        "Missing essential claim",
    ],
)

# Use the selected dashboard transaction.
# Keep full score precision for consistent decisions.
exact_score = float(
    artifact["model"].predict_proba(
        pd.DataFrame([transaction])[artifact["features"]]
    )[0, 1]
)

trusted_prediction = {
    "fraud_score": exact_score,
    "threshold": threshold,
    "flagged": exact_score >= threshold,
}

evidence = build_investigation_evidence(
    transaction,
    trusted_prediction,
)

claims = [
    {
        "field": "model_score",
        "value": evidence["model_score"],
    },
    {
        "field": "review_required",
        "value": evidence["review_required"],
    },
]

if scenario == "Unsupported fraud accusation":
    claims.append({
        "field": "confirmed_fraud",
        "value": True,
    })

elif scenario == "Fabricated location":
    claims.append({
        "field": "country",
        "value": "France",
    })

elif scenario == "Missing essential claim":
    claims = [
        {
            "field": "model_score",
            "value": evidence["model_score"],
        }
    ]

if st.button("Evaluate AI Claims"):
    review = build_safe_response(
        evidence,
        claims,
    )

    st.subheader("Safety Evaluation")

    st.write(f"**Result:** {review['status'].upper()}")

    st.metric(
        "Rejected Claims",
        review["rejected_count"],
    )

    if review["status"] == "approved":
        st.success("All submitted claims were supported.")

    elif review["status"] == "filtered":
        st.warning(
            "Unsupported claims were removed. "
            "A safe summary was reconstructed."
        )

    else:
        st.info(
            "Essential claims were missing. "
            "The trusted reference summary was used."
        )

    st.subheader("Investigator-Facing Summary")
    st.text(review["text"])

st.caption(
    "This is a controlled demonstration using simulated "
    "structured claims, not a live generative AI model."
)
