"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:
    python -m src.monitoring
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import config as cfg

from evidently import Report
from evidently.presets import DataDriftPreset


# ---------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------

def calculate_psi(reference_scores, current_scores, bins=10):
    reference_scores = np.asarray(reference_scores, dtype=float)
    current_scores = np.asarray(current_scores, dtype=float)

    # Create bins using the reference distribution
    breakpoints = np.quantile(
        reference_scores,
        np.linspace(0, 1, bins + 1)
    )

    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    # Need at least 2 boundaries
    if len(breakpoints) < 2:
        return 0.0

    # Extend the first/last boundary
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_scores,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current_scores,
        bins=breakpoints
    )

    # Convert counts to proportions
    reference_pct = reference_counts / len(reference_scores)
    current_pct = current_counts / len(current_scores)

    # Avoid log(0)
    epsilon = 1e-6

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ---------------------------------------------------------
# PSI interpretation
# ---------------------------------------------------------

def interpret_psi(psi):
    """
    Standard PSI interpretation.
    """

    if psi < 0.10:
        return "No significant prediction drift"
    elif psi < 0.25:
        return "Moderate prediction drift"
    else:
        return "Significant prediction drift"


# ---------------------------------------------------------
# Main monitoring function
# ---------------------------------------------------------

def run_monitoring():

    print("=" * 60)
    print("STAGE 4.4 — MODEL MONITORING")
    print("=" * 60)

    artifacts_dir = Path("artifacts")

    reference_path = artifacts_dir / "reference_sample.csv"
    current_path = artifacts_dir / "current_batch.csv"

    drift_report_path = artifacts_dir / "drift_report.html"
    summary_path = artifacts_dir / "drift_summary.json"

    # -----------------------------------------------------
    # 4.4.1 — Generate current batch
    # -----------------------------------------------------

    print("\n[1] Generating current batch...")

    if not current_path.exists():

        result = subprocess.run(
            [sys.executable, "-m", "src.generate_current_batch"],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(
                "Current batch generation failed."
            )

    if not current_path.exists():
        raise FileNotFoundError(
            f"Current batch not found: {current_path}"
        )

    # -----------------------------------------------------
    # Load reference and current data
    # -----------------------------------------------------

    print("\n[2] Loading reference and current data...")

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    print("Reference shape:", reference.shape)
    print("Current shape:", current.shape)

    # Keep only columns available in both datasets
    common_columns = [
        column
        for column in reference.columns
        if column in current.columns
    ]

    reference_features = reference[common_columns].copy()
    current_features = current[common_columns].copy()

    print("Common feature columns:", len(common_columns))

    # -----------------------------------------------------
    # 4.4.1 — Evidently feature drift
    # -----------------------------------------------------

    print("\n[3] Running Evidently feature drift...")

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        reference_data=reference_features,
        current_data=current_features
    )

    snapshot.save_html(str(drift_report_path))

    print("Evidently report saved to:")
    print(drift_report_path)

    # Extract drift summary from Evidently
    report_dict = snapshot.dict()

    number_of_drifted_columns = None
    number_of_columns = len(common_columns)
    dataset_drift = None

    # Search Evidently's output for drift metrics
    for metric in report_dict.get("metrics", []):

        metric_id = metric.get("metric_id", "")
        value = metric.get("value")

        if "DriftedColumns" in metric_id:
            if isinstance(value, dict):
                number_of_drifted_columns = value.get(
                    "count",
                    value.get("drifted_columns")
                )

        if "DriftedColumnsCount" in metric_id:
            number_of_drifted_columns = value

        if "DatasetDrift" in metric_id:
            if isinstance(value, dict):
                dataset_drift = value.get(
                    "dataset_drift",
                    value.get("drift_detected")
                )
            elif isinstance(value, bool):
                dataset_drift = value

    # Fallback if Evidently's internal structure differs
    if number_of_drifted_columns is None:

        number_of_drifted_columns = 0

        for metric in report_dict.get("metrics", []):

            value = metric.get("value", {})

            if isinstance(value, dict):

                if "count" in value and "share" in value:
                    number_of_drifted_columns = value["count"]

                if "dataset_drift" in value:
                    dataset_drift = value["dataset_drift"]

    print("\nFeature Drift Summary")
    print("---------------------")
    print(
        "Drifted columns:",
        number_of_drifted_columns
    )
    print(
        "Total columns:",
        number_of_columns
    )
    print(
        "Dataset drift:",
        dataset_drift
    )

    # -----------------------------------------------------
    # 4.4.2 — Prediction drift / PSI
    # -----------------------------------------------------

    print("\n[4] Calculating prediction drift...")

    model_path = artifacts_dir / "best_model.pkl"

    model = joblib.load(model_path)

    # The model expects exactly the columns it was trained with.
    # Use the saved input column list when available.
    input_columns_path = artifacts_dir / "input_columns.json"

    if input_columns_path.exists():

        with open(input_columns_path, "r") as f:
            input_columns = json.load(f)

    else:
        input_columns = common_columns

    # Add missing model input columns as NaN.
    # The trained pipeline's imputer handles them.
    reference_model_data = reference.copy()
    current_model_data = current.copy()

    for column in input_columns:

        if column not in reference_model_data.columns:
            reference_model_data[column] = np.nan

        if column not in current_model_data.columns:
            current_model_data[column] = np.nan

    reference_model_data = reference_model_data[input_columns]
    current_model_data = current_model_data[input_columns]

    # Predict probabilities
    reference_scores = model.predict_proba(
        reference_model_data
    )[:, 1]

    current_scores = model.predict_proba(
        current_model_data
    )[:, 1]

    prediction_psi = calculate_psi(
        reference_scores,
        current_scores
    )

    psi_interpretation = interpret_psi(prediction_psi)

    print("\nPrediction Drift")
    print("----------------")
    print(
        f"Prediction PSI: {prediction_psi:.4f}"
    )
    print(
        f"Interpretation: {psi_interpretation}"
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary = {
        "feature_drift": {
            "number_of_drifted_columns": number_of_drifted_columns,
            "number_of_columns": number_of_columns,
            "dataset_drift": dataset_drift
        },
        "prediction_drift": {
            "psi": round(prediction_psi, 4),
            "interpretation": psi_interpretation
        }
    }

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n[5] Monitoring summary saved to:")
    print(summary_path)

    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print("=" * 60)

    return summary


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_monitoring()
