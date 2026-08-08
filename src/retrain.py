"""
retrain.py — Stage 4.5: multi-signal retraining trigger + workflow.

Run:
    python -m src.retrain
"""

import json
from pathlib import Path

import config as cfg


# ============================================================
# Retraining thresholds
# ============================================================

PSI_THRESHOLD = 0.20
DRIFT_SHARE_THRESHOLD = 0.30


# ============================================================
# 4.5.1 — Multi-signal retraining trigger
# ============================================================

def decide(summary):
    """
    Return a list of human-readable reasons if any retraining
    trigger fires.

    Triggers:
        - prediction PSI > 0.20
        - drifted feature share > 0.30
        - dataset drift == True

    An empty list means no retraining is required.
    """

    reasons = []

    prediction_psi = float(summary.get("prediction_psi", 0.0))
    share_drifted = float(summary.get("share_drifted", 0.0))
    dataset_drift = bool(summary.get("dataset_drift", False))

    if prediction_psi > PSI_THRESHOLD:
        reasons.append(
            f"Prediction PSI {prediction_psi:.3f} > threshold {PSI_THRESHOLD:.2f}"
        )

    if share_drifted > DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"Drifted feature share {share_drifted:.3f} > threshold "
            f"{DRIFT_SHARE_THRESHOLD:.2f}"
        )

    if dataset_drift:
        reasons.append(
            "Dataset drift detected"
        )

    return reasons


# ============================================================
# 4.5.2 — Retraining workflow
# ============================================================

def run_retraining_workflow():

    summary_path = Path("artifacts/drift_summary.json")
    decision_path = Path("artifacts/retraining_decision.json")

    # --------------------------------------------------------
    # Load monitoring results
    # --------------------------------------------------------

    if not summary_path.exists():
        raise FileNotFoundError(
            "artifacts/drift_summary.json was not found. "
            "Run Stage 4.4 monitoring first."
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("Loaded drift summary:")
    print(json.dumps(summary, indent=4))

    # --------------------------------------------------------
    # Evaluate retraining trigger
    # --------------------------------------------------------

    reasons = decide(summary)

    # --------------------------------------------------------
    # No retraining required
    # --------------------------------------------------------

    if not reasons:

        decision = {
            "retrain": False,
            "reasons": [],
            "message": "No retraining trigger fired."
        }

        with open(decision_path, "w") as f:
            json.dump(decision, f, indent=4)

        print("\nRetraining NOT required.")
        print(json.dumps(decision, indent=4))

        return decision

    # --------------------------------------------------------
    # Retraining required
    # --------------------------------------------------------

    print("\nRetraining triggered!")
    print("Reasons:")

    for reason in reasons:
        print(f"- {reason}")

    # Import here so that training is only loaded when needed.
    from src.train import train_and_log

    print("\nStarting retraining workflow...")

    # train_and_log() performs the training, MLflow logging,
    # model registration and production promotion.
    new_metrics = train_and_log()

    # --------------------------------------------------------
    # Record retraining decision
    # --------------------------------------------------------

    decision = {
        "retrain": True,
        "reasons": reasons,
        "new_metrics": new_metrics
    }

    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=4, default=str)

    print("\nRetraining decision:")
    print(json.dumps(decision, indent=4))

    return decision


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_retraining_workflow()