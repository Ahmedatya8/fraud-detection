# IEEE-CIS Fraud Detection API

> ML service that detects fraudulent transactions using real Vesta Corporation
> transaction data from the IEEE-CIS Fraud Detection Kaggle competition.
> Served via REST API, containerized with Docker, deployed on Render.

🔗 **Live API:** https://fraud-detection-muto.onrender.com/docs

---

## Problem

Financial fraud costs billions annually. Traditional rule-based systems
miss sophisticated fraud patterns. This service uses machine learning
trained on 590,000 real transactions to flag fraud in real time,
returning a probability score and risk level for every transaction.

---

## Dataset

- **Source:** IEEE-CIS Fraud Detection (Kaggle Competition)
- **Provider:** Vesta Corporation — real production transaction data
- **Size:** 590,000 transactions, 434 features across 2 tables
- **Fraud rate:** 3.5% — severe class imbalance

---

## The Core Challenge — Class Imbalance

A model predicting "not fraud" on everything scores 96.5% accuracy
but catches zero fraud cases. This is why accuracy is useless here.

**We optimize for:**
| Metric | What it measures |
|---|---|
| Recall | % of fraud cases actually caught |
| Precision | % of fraud flags that are real fraud |
| F1 | Balance of precision and recall |
| AUC-ROC | Overall discrimination ability |

---

## Model & Results

3 models compared via Stratified 3-fold cross-validation:

| Model | CV AUC | CV F1 |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.52 |
| Random Forest | ~0.91 | ~0.71 |
| **XGBoost** | **~0.95** | **~0.78** |

**Final model: [YOUR BEST MODEL]**

| Metric | Value |
|---|---|
| AUC-ROC | [YOUR VALUE] |
| F1 Score | [YOUR VALUE] |
| Precision | [YOUR VALUE] |
| Recall | [YOUR VALUE] |

**Imbalance strategy:** `scale_pos_weight` for XGBoost /
`class_weight='balanced'` for other models — no oversampling needed.

---

## API Usage

**Base URL:** `https://fraud-detection-muto.onrender.com`

### Health check
```bash
curl https://fraud-detection-muto.onrender.com/health
```

### Predict fraud
```bash
curl -X POST https://fraud-detection-muto.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.0,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "day": 10,
    "hour": 14
  }'
```

### Response
```json
{
  "fraud_probability": 0.0312,
  "prediction": 0,
  "prediction_label": "LEGITIMATE",
  "risk_level": "LOW",
  "input_received": { ... }
}
```

---

## How to Run Locally

```bash
# Clone
git clone https://github.com/Ahmedatya8/fraud-detection
cd fraud-detection

# Install
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload

# Open docs
http://127.0.0.1:8000/docs
```

---

## Run with Docker

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

---

## Project Structure
fraud-detection/

├── notebooks/

│   ├── 01_eda.ipynb          # class imbalance, missing values, fraud patterns

│   └── 02_train.ipynb        # pipeline, model comparison, evaluation

├── src/

│   └── predict.py            # prediction logic

├── api/

│   └── main.py               # FastAPI app

├── models/

│   └── model_meta.json       # metrics and feature info

├── Dockerfile

└── requirements.txt

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `imbalanced-learn` ·
`FastAPI` · `Docker` · `Render`