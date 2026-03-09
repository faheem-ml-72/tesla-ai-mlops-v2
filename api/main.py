from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import pandas as pd
import logging
import tensorflow as tf
from pydantic import BaseModel
from typing import List

# -------------------------------
# Logging configuration
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------------------
# Create FastAPI app
# -------------------------------

app = FastAPI(title="Tesla AI Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load models at startup
# -------------------------------

try:
    logger.info("Loading models...")

    lstm_model = tf.keras.layers.TFSMLayer(
    "models/lstm_saved_model",
    call_endpoint="serve"
)
    gp_model = joblib.load("models/gp_uncertainty_model.pkl")
    xgb_model = joblib.load("models/xgb_tesla_model.pkl")

    logger.info("Models loaded successfully")

except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise e


# -------------------------------
# Request schema
# -------------------------------

class InputData(BaseModel):
    data: List[List[float]]


# -------------------------------
# Root endpoint
# -------------------------------

@app.get("/")
def home():
    return {"message": "Tesla AI Forecast API running"}


# -------------------------------
# Health check endpoint
# -------------------------------

@app.get("/health")
def health():
    return {"status": "API is healthy"}


# -------------------------------
# Prediction endpoint
# -------------------------------

@app.post("/predict")
def predict(input_data: InputData):

    try:
        logger.info("Prediction request received")

        arr = np.array(input_data.data)

        # Validate input
        if arr.shape != (30, 5):
            raise HTTPException(
                status_code=400,
                detail="Input must be shape (30,5)"
            )

        # Reshape for LSTM
        X_input = arr.reshape(1, 30, 5)

        # LSTM prediction
        lstm_pred = lstm_model.predict(X_input, verbose=0)[0][0]

        # XGBoost prediction
        xgb_input = X_input.reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_input)[0]

        # Ensemble prediction
        ensemble_pred = 0.6 * lstm_pred + 0.4 * xgb_pred

        # Gaussian Process uncertainty
        _, gp_std = gp_model.predict([[ensemble_pred]], return_std=True)

        lower_bound = ensemble_pred - 2 * gp_std[0]
        upper_bound = ensemble_pred + 2 * gp_std[0]

        # Risk level classification
        if gp_std[0] < 0.02:
            risk = "Low"
        elif gp_std[0] < 0.05:
            risk = "Moderate"
        else:
            risk = "High"

        # Save prediction history
        prediction_log = pd.DataFrame([{
            "prediction": float(ensemble_pred),
            "uncertainty": float(gp_std[0]),
            "risk": risk
        }])

        prediction_log.to_csv(
            "api/data/predictions.csv",
            mode="a",
            header=False,
            index=False
        )

        logger.info("Prediction completed successfully")

        return {
            "forecast_price": float(ensemble_pred),
            "confidence_interval": [
                float(lower_bound),
                float(upper_bound)
            ],
            "risk_level": risk,
            "uncertainty": float(gp_std[0])
        }

    except Exception as e:

        logger.error(f"Prediction failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )