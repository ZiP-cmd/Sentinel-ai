
MODEL_NAME = "Weighted Logistic Regression"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20

RANDOM_STATE = 42

MODEL_PARAMETERS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# Fixed review capacity for the final evaluation batch.
EVALUATION_ALERT_BUDGET = 100

# Record the selected policy before inspecting test labels.
MODEL_SELECTION_FROZEN = True
