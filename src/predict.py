# ── predict.py ────────────────────────────────────────────────────────────────
# Prediction logic isolated from the API layer.
# Loads the trained pipeline once and reuses it on every request.

import numpy as np
import joblib
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'model.joblib'
META_PATH  = ROOT / 'models' / 'model_meta.json'

# ── Lazy loading ──────────────────────────────────────────────────────────────
# Load model once at first prediction — not on every request.
_pipeline = None
_meta     = None


def _load():
    """Load pipeline and metadata from disk if not already loaded."""
    global _pipeline, _meta
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)


def predict_fraud(
    TransactionAmt: float,
    ProductCD: str,
    card1: float,
    card2: float,
    card3: float,
    card4: str,
    card5: float,
    card6: str,
    addr1: float,
    addr2: float,
    dist1: float,
    P_emaildomain: str,
    R_emaildomain: str,
    C1: float, C2: float, C3: float,
    C4: float, C5: float, C6: float,
    V1: float, V2: float, V3: float,
    V4: float, V5: float,
    day: int,
    hour: int,
) -> dict:
    """
    Accept raw transaction features, run through saved pipeline,
    return fraud probability and prediction.
    """
    _load()

    import pandas as pd

    # ── Build input DataFrame ─────────────────────────────────────────────────
    # Must match exact feature names and order used during training.
    X = pd.DataFrame([{
        'TransactionAmt':     TransactionAmt,
        'log_TransactionAmt': np.log1p(TransactionAmt),
        'day':                day,
        'hour':               hour,
        'card1':              card1,
        'card2':              card2,
        'card3':              card3,
        'card5':              card5,
        'addr1':              addr1,
        'addr2':              addr2,
        'dist1':              dist1,
        'C1': C1, 'C2': C2, 'C3': C3,
        'C4': C4, 'C5': C5, 'C6': C6,
        'V1': V1, 'V2': V2, 'V3': V3,
        'V4': V4, 'V5': V5,
        'ProductCD':     ProductCD,
        'card4':         card4,
        'card6':         card6,
        'P_emaildomain': P_emaildomain,
        'R_emaildomain': R_emaildomain,
    }])

    # ── Filter to only columns the model was trained on ───────────────────────
    trained_cols = _meta['features']['numeric'] + _meta['features']['categorical']
    X = X[[c for c in trained_cols if c in X.columns]]

    # ── Predict ───────────────────────────────────────────────────────────────
    fraud_prob = float(_pipeline.predict_proba(X)[0][1])
    prediction = int(_pipeline.predict(X)[0])

    # ── Risk level ───────────────────────────────────────────────────────────
    if fraud_prob >= 0.7:
        risk = "HIGH"
    elif fraud_prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        'fraud_probability': round(fraud_prob, 4),
        'prediction':        prediction,           # 1 = fraud, 0 = legitimate
        'prediction_label':  'FRAUD' if prediction == 1 else 'LEGITIMATE',
        'risk_level':        risk,
    }