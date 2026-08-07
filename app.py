"""
app.py — Stage 4.1: FastAPI inference service.
Implement GET /health and POST /predict, then demonstrate it from the operations notebook.

Run:  uvicorn app:app --reload
"""

# TODO (Stage 4.1):
# - Lazily load artifacts/best_model.pkl and artifacts/input_columns.json.
# - GET  /health  -> {"status": ..., "model_loaded": bool}.
# - POST /predict -> align the incoming features to the training columns (missing -> NaN; the
#   pipeline imputes), return {"readmission_probability", "readmitted_30d", "threshold"}.
#   Validate the payload (reject an empty features dict with 422) and log every prediction to
#   artifacts/predictions.log (governance). Note: JSON has no NaN, so clients send null / omit fields.
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# Configuration
# =========================================================

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

MODEL_PATH = ARTIFACTS / "best_model.pkl"
COLUMNS_PATH = ARTIFACTS / "input_columns.json"
LOG_PATH = ARTIFACTS / "predictions.log"

THRESHOLD = 0.5


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Hospital Readmission Predictor",
    version="1.0"
)


# =========================================================
# Request validation
# =========================================================

class PredictRequest(BaseModel):
    features: dict = Field(
        ...,
        description="Raw patient encounter features"
    )


# =========================================================
# Lazy-loaded artifacts
# =========================================================

_model = None
_input_columns = None


def load_artifacts():

    global _model
    global _input_columns

    # Load model only when first prediction is requested
    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    # Load input columns only when first prediction is requested
    if _input_columns is None:

        if not COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Input columns file not found: {COLUMNS_PATH}"
            )

        with open(COLUMNS_PATH, "r") as f:
            _input_columns = json.load(f)

    return _model, _input_columns


# =========================================================
# GET /health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists()
    }


# =========================================================
# POST /predict
# =========================================================

@app.post("/predict")
def predict(request: PredictRequest):

    # -----------------------------------------------------
    # Validate that features were supplied
    # -----------------------------------------------------

    if not request.features:

        raise HTTPException(
            status_code=422,
            detail="Features dictionary cannot be empty."
        )

    try:

        # -------------------------------------------------
        # Load model and training columns
        # -------------------------------------------------

        model, input_columns = load_artifacts()

        # -------------------------------------------------
        # Convert incoming JSON to DataFrame
        # -------------------------------------------------

        df = pd.DataFrame([request.features])

        # -------------------------------------------------
        # Align input columns with training columns
        #
        # Missing columns are filled with NaN.
        # Extra columns are ignored.
        # -------------------------------------------------

        for column in input_columns:

            if column not in df.columns:
                df[column] = np.nan

        df = df[input_columns]

        # -------------------------------------------------
        # Convert pandas missing values to numpy NaN
        # -------------------------------------------------

        df = df.replace({pd.NA: np.nan})

        # -------------------------------------------------
        # Generate prediction
        # -------------------------------------------------

        probability = float(
            model.predict_proba(df)[0][1]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        # -------------------------------------------------
        # Build response
        # -------------------------------------------------

        response = {
            "readmission_probability": round(
                probability,
                4
            ),
            "readmitted_30d": prediction,
            "threshold": THRESHOLD
        }

        # -------------------------------------------------
        # Log prediction
        # -------------------------------------------------

        ARTIFACTS.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(LOG_PATH, "a") as f:

            f.write(
                json.dumps(response) + "\n"
            )

        return response

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )