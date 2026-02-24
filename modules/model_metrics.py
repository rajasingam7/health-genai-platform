import json
import os

METRICS_PATH = "models/model_metrics.json"


def load_model_metrics():
    """
    Load saved model performance metrics.
    Returns None if file does not exist.
    """

    if not os.path.exists(METRICS_PATH):
        return None

    with open(METRICS_PATH, "r") as f:
        return json.load(f)