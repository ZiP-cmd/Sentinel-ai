
# Sentinel AI — Model Selection

## Selected Model
Class-weighted Logistic Regression with StandardScaler.

## Data Splitting
Chronological split:
- Training: 60%
- Validation: 20%
- Test: 20%

The test set was not used for model selection.

## Model Selection Rationale
The class-weighted logistic regression model achieved
the highest validation average precision among the
three evaluated model families.

## Selected Evaluation Policy
Top 100 highest-scoring transactions in the
held-out test batch.

This is an evaluation policy, not a production
daily investigation budget.

## Limitations
- Fraud cases are extremely rare.
- The dataset is historical and anonymized.
- Scores are not guaranteed to be calibrated.
- Financial costs used in the dashboard are hypothetical.
- The model has not been validated on live transactions.
- The evaluation budget may not generalize to other batch sizes.

## Final Held-Out Evaluation

**Status:** Completed
**Selected model:** Class-weighted Logistic Regression with StandardScaler
**Model selection:** Based on chronological validation data
**Model artifact:** Frozen and verified using SHA-256
**Evaluation policy:** Top 100 highest-scoring transactions across the full test partition

### Test Dataset

- Total transactions: 56,962
- Actual fraud cases: 75
- Chronological split: Final 20% of the original dataset

### Final Model Performance

| Metric | Result |
|---|---:|
| Average precision | 0.7436 |
| Precision at 100 alerts | 0.5900 |
| Recall at 100 alerts | 0.7867 |
| True positives | 59 |
| False positives | 41 |
| False negatives | 16 |
| True negatives | 56,846 |

### Interpretation

The selected model detected 59 of 75 fraudulent transactions within the 100 highest-scoring transactions in the held-out test partition.

Of the 100 transactions selected for investigation, 59 were genuinely fraudulent and 41 were legitimate.

The model maintained recall close to its validation performance, while average precision decreased from 0.7719 to 0.7436.

### Evaluation Integrity

The model and investigation budget were selected before final test evaluation. The saved model artifact's SHA-256 hash was verified before test predictions were generated.

The final test results were not used to retrain the model, change its configuration, or select a different investigation budget.

### Limitations

- The dataset is historical, highly imbalanced, and anonymized.
- The review budget applies retrospectively to an entire test partition, not to a live daily workflow.
- Scores have not been demonstrated to be calibrated probabilities.
- Financial assumptions used in the dashboard are hypothetical.
- The model has not been validated on live transaction streams.
- Although the model architecture and dataset boundaries were verified, the exact original training-row lineage was not independently established.
- Performance may change across time periods, financial institutions, and fraud patterns.

### Conclusion

The frozen weighted logistic regression model demonstrated useful fraud-ranking performance on the held-out historical test partition.

The results establish a documented offline evaluation baseline. They do not constitute production deployment approval.

## Frozen Model Artifact

File: models/sentinel_fraud_model.joblib

SHA-256:
4ABC7B3024B3301BF9525D086259905D69EC7383778BF08A85514AB7A86F9842

The model artifact was fingerprinted before final
test-set evaluation.