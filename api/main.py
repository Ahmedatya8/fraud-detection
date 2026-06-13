# ── main.py ───────────────────────────────────────────────────────────────────
# FastAPI application — public interface to the fraud detection model.
# Validates requests, calls predict_fraud(), returns structured response.

import sys
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_fraud

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title='IEEE-CIS Fraud Detection API',
    description='Detects fraudulent transactions using a model trained on real Vesta Corporation data.',
    version='1.0.0',
)


# ── Request schema ────────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    TransactionAmt: float  = Field(..., gt=0,    example=150.0,
                                   description='Transaction amount in USD')
    ProductCD:      str    = Field(...,           example='W',
                                   description='Product code: W, H, C, S, R')
    card1:          float  = Field(0.0,           example=9500.0)
    card2:          float  = Field(0.0,           example=111.0)
    card3:          float  = Field(0.0,           example=150.0)
    card4:          str    = Field('visa',        example='visa',
                                   description='Card network: visa, mastercard, etc.')
    card5:          float  = Field(0.0,           example=226.0)
    card6:          str    = Field('debit',       example='debit',
                                   description='Card type: debit or credit')
    addr1:          float  = Field(0.0,           example=315.0)
    addr2:          float  = Field(0.0,           example=87.0)
    dist1:          float  = Field(0.0,           example=0.0)
    P_emaildomain:  str    = Field('gmail.com',   example='gmail.com')
    R_emaildomain:  str    = Field('gmail.com',   example='gmail.com')
    C1:             float  = Field(0.0,           example=1.0)
    C2:             float  = Field(0.0,           example=1.0)
    C3:             float  = Field(0.0,           example=0.0)
    C4:             float  = Field(0.0,           example=0.0)
    C5:             float  = Field(0.0,           example=0.0)
    C6:             float  = Field(0.0,           example=1.0)
    V1:             float  = Field(0.0,           example=1.0)
    V2:             float  = Field(0.0,           example=1.0)
    V3:             float  = Field(0.0,           example=1.0)
    V4:             float  = Field(0.0,           example=1.0)
    V5:             float  = Field(0.0,           example=1.0)
    day:            int    = Field(0, ge=0,       example=10,
                                   description='Day number from dataset reference date')
    hour:           int    = Field(0, ge=0, le=23, example=14,
                                   description='Hour of day (0-23)')


# ── Response schema ───────────────────────────────────────────────────────────
class FraudPrediction(BaseModel):
    fraud_probability: float
    prediction:        int
    prediction_label:  str
    risk_level:        str
    input_received:    dict


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    """Health check — used by deployment platforms."""
    return {'status': 'ok', 'model': 'ieee-cis-fraud-detector-v1'}


@app.post('/predict', response_model=FraudPrediction)
def predict(data: TransactionInput):
    """
    Predict whether a transaction is fraudulent.

    Returns:
    - fraud_probability: 0.0 to 1.0
    - prediction: 0 (legitimate) or 1 (fraud)
    - prediction_label: LEGITIMATE or FRAUD
    - risk_level: LOW / MEDIUM / HIGH
    """
    try:
        result = predict_fraud(
            TransactionAmt=data.TransactionAmt,
            ProductCD=data.ProductCD,
            card1=data.card1,
            card2=data.card2,
            card3=data.card3,
            card4=data.card4,
            card5=data.card5,
            card6=data.card6,
            addr1=data.addr1,
            addr2=data.addr2,
            dist1=data.dist1,
            P_emaildomain=data.P_emaildomain,
            R_emaildomain=data.R_emaildomain,
            C1=data.C1, C2=data.C2, C3=data.C3,
            C4=data.C4, C5=data.C5, C6=data.C6,
            V1=data.V1, V2=data.V2, V3=data.V3,
            V4=data.V4, V5=data.V5,
            day=data.day,
            hour=data.hour,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {**result, 'input_received': data.model_dump()}