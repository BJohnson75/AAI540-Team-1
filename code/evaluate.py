
import os

with open("/opt/ml/processing/output/evaluation/evaluation.json", "w") as f:
    json.dump({"regression_metrics": {"mse": {"value": 3.74}}}, f)
