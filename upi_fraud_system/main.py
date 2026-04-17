"""
main.py — FastAPI Backend
=========================
Endpoints:
  POST /predict        analyse a transaction
  GET  /transactions   retrieve recent logs
  GET  /analytics      aggregated dashboard stats
  GET  /health         system health check

SIEM: posts JSON events to Logstash on localhost:5000
SMS : Twilio alert for HIGH-risk transactions
DB  : MongoDB (pymongo) — falls back to in-memory if unavailable
"""

import os, uuid, logging
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Optional integrations (graceful fallback) ─────────
try:
    import requests as _req
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from pymongo import MongoClient
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    _client.server_info()
    db         = _client["upi_fraud"]
    txn_col    = db["transactions"]
    MONGO_OK   = True
    print("✅ MongoDB connected")
except Exception:
    MONGO_OK   = False
    _mem_store = []               # in-memory fallback
    print("⚠️  MongoDB not available — using in-memory store")
"""
try:
    from twilio.rest import Client as TwilioClient
    _twilio = TwilioClient(
        os.getenv("TWILIO_SID", ""),
        os.getenv("TWILIO_TOKEN", "")
    )
    TWILIO_OK = bool(os.getenv("TWILIO_SID"))
    print("✅ Twilio configured" if TWILIO_OK else "⚠️  Twilio env vars not set")
except Exception:
    TWILIO_OK = False
"""
"""
try:
    from twilio.rest import Client as TwilioClient

    TWILIO_SID   = os.getenv("TWILIO_SID")
    TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")

    if TWILIO_SID and TWILIO_TOKEN:
        _twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        TWILIO_OK = True
        print("✅ Twilio configured")
    else:
        TWILIO_OK = False
        print("⚠️ Twilio env vars not set")

except Exception:
    TWILIO_OK = False
    print("⚠️ Twilio initialization failed")
"""
# ── Twilio (FINAL FIXED VERSION) ─────────
from dotenv import load_dotenv
load_dotenv()   # ✅ VERY IMPORTANT

try:
    from twilio.rest import Client as TwilioClient

    TWILIO_SID    = os.getenv("TWILIO_SID")
    TWILIO_TOKEN  = os.getenv("TWILIO_TOKEN")
    TWILIO_FROM   = os.getenv("TWILIO_FROM")
    TWILIO_ADMIN  = os.getenv("TWILIO_ADMIN")

    if all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_ADMIN]):
        _twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        TWILIO_OK = True
        print("✅ Twilio fully configured")
    else:
        TWILIO_OK = False
        print("❌ Missing Twilio env variables")

except Exception as e:
    TWILIO_OK = False
    print(f"❌ Twilio init error: {e}")
# ── Load models ────────────────────────────────────────
MODEL_DIR = "models"
try:
    rf           = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    le_location  = joblib.load(f"{MODEL_DIR}/le_location.pkl")
    le_device    = joblib.load(f"{MODEL_DIR}/le_device.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    MODELS_OK    = True
    print("✅ Models loaded")
except Exception as e:
    MODELS_OK    = False
    print(f"⚠️  Models not found ({e}) — run train_model.py first")

# ── Risk engine import ────────────────────────────────
from risk_engine import compute_risk_score

# ── FastAPI app ────────────────────────────────────────
app = FastAPI(title="UPI FraudShield API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Serve dashboard.html at /
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_dashboard():
    if os.path.exists("static/dashboard.html"):
        return FileResponse("static/dashboard.html")
    return {"message": "UPI FraudShield API — see /docs"}

# ══════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════
class TransactionIn(BaseModel):
    sender_upi          : str   = Field(..., example="rahul@okaxis")
    receiver_upi        : str   = Field(..., example="shop@ybl")
    amount              : float = Field(..., example=5000.0)
    payment_type        : str   = Field("P2P",  example="P2P")
    hour_of_day         : int   = Field(14,      example=14)
    day_of_week         : int   = Field(1,       example=1)
    device_type         : str   = Field(...,     example="iPhone_14")
    location            : str   = Field(...,     example="Mumbai")
    is_new_receiver     : int   = Field(0,       example=0)
    txn_frequency_1hr   : int   = Field(2,       example=2)
    failed_pin_attempts : int   = Field(0,       example=0)
    vpn_used            : int   = Field(0,       example=0)

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def _safe_encode(encoder, value: str, default=0) -> int:
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return default

def _build_feature_vector(txn: TransactionIn) -> np.ndarray:
    location_enc    = _safe_encode(le_location, txn.location)
    device_type_enc = _safe_encode(le_device,   txn.device_type)

    is_odd_hour       = int(txn.hour_of_day < 6 or txn.hour_of_day > 22)
    is_large_amount   = int(txn.amount > 50_000)
    is_high_frequency = int(txn.txn_frequency_1hr > 10)
    risk_device       = int(any(k in txn.device_type.lower()
                                for k in ["unknown","emulator","rooted","vm"]))
    risk_location     = int(any(k in txn.location.lower()
                                for k in ["unknown","foreign","spoofed","tor","vpn"]))

    row = {
        "amount"              : txn.amount,
        "hour_of_day"         : txn.hour_of_day,
        "day_of_week"         : txn.day_of_week,
        "is_new_receiver"     : txn.is_new_receiver,
        "txn_frequency_1hr"   : txn.txn_frequency_1hr,
        "failed_pin_attempts" : txn.failed_pin_attempts,
        "vpn_used"            : txn.vpn_used,
        "location_enc"        : location_enc,
        "device_type_enc"     : device_type_enc,
        "is_odd_hour"         : is_odd_hour,
        "is_large_amount"     : is_large_amount,
        "is_high_frequency"   : is_high_frequency,
        "risk_device"         : risk_device,
        "risk_location"       : risk_location,
    }
    return np.array([[row[c] for c in feature_cols]])

def _post_to_logstash(event: dict):
    if not REQUESTS_OK:
        return
    try:
        _req.post("http://localhost:5000", json=event, timeout=1)
    except Exception:
        pass   # Logstash not running — silent fail
"""
def _send_sms(txn_id: str, amount: float, risk_score: float, sender: str):
    if not TWILIO_OK:
        print(f"  [SMS MOCK] HIGH RISK: {txn_id} | ₹{amount} | score={risk_score}")
        return
    try:
        _twilio.messages.create(
            body=(f"🚨 HIGH RISK FRAUD ALERT\n"
                  f"TXN: {txn_id}\nAmount: ₹{amount:,.2f}\n"
                  f"Risk Score: {risk_score}\nUser: {sender}"),
            from_=os.getenv("TWILIO_FROM", "+1234567890"),
            to=os.getenv("TWILIO_ADMIN", "+0000000000"),
        )
    except Exception as e:
        print(f"  [SMS ERROR] {e}")
"""

"""
def _send_sms(txn_id, amount, risk_score, sender):
    if not TWILIO_OK:
        print(f"[SMS MOCK] {txn_id}")
        return

    try:
        _twilio.messages.create(
            body=f"🚨 FRAUD ALERT\nTXN: {txn_id}\n₹{amount}\nScore: {risk_score}",
            from_=os.getenv("TWILIO_FROM"),
            to=os.getenv("TWILIO_ADMIN"),
        )
    except Exception as e:
        print(f"[SMS ERROR] {e}")
"""
def _send_sms(txn_id, amount, risk_score, sender):
    if not TWILIO_OK:
        print(f"[SMS MOCK] {txn_id}")
        return

    try:
        message = _twilio.messages.create(
            body=(
                f"🚨 UPI FRAUD ALERT 🚨\n"
                f"Transaction: {txn_id}\n"
                f"Amount: ₹{amount}\n"
                f"Risk Score: {risk_score}\n"
                f"User: {sender}"
            ),
            from_=TWILIO_FROM,
            to=TWILIO_ADMIN,
        )

        print(f"✅ SMS SENT: SID = {message.sid}")

    except Exception as e:
        print(f"❌ SMS FAILED: {e}")

def _save_record(record: dict):
    if MONGO_OK:
        txn_col.insert_one({k: v for k, v in record.items() if k != "_id"})
    else:
        _mem_store.append(record)
        if len(_mem_store) > 500:
            _mem_store.pop(0)

def _get_records(limit=50):
    if MONGO_OK:
        return list(txn_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return list(reversed(_mem_store[-limit:]))

# ══════════════════════════════════════════════════════
# POST /predict
# ══════════════════════════════════════════════════════
@app.post("/predict")
async def predict(txn: TransactionIn):
    if not MODELS_OK:
        raise HTTPException(503, "Models not loaded — run train_model.py first")

    txn_id    = "TXN" + uuid.uuid4().hex[:10].upper()
    timestamp = datetime.utcnow().isoformat() + "Z"

    # ── Feature vector ─────────────────────────────
    X = _build_feature_vector(txn)

    # ── Model inference ────────────────────────────
    fraud_probability = float(rf.predict_proba(X)[0][1])

    # ── Risk scoring engine ────────────────────────
    risk_result = compute_risk_score(fraud_probability, txn.dict())
    risk_score  = risk_result["risk_score"]
    risk_level  = risk_result["risk_level"]

    # ── SIEM event to Logstash ─────────────────────
    siem_event = {
        "transaction_id"   : txn_id,
        "timestamp"        : timestamp,
        "sender_upi"       : txn.sender_upi,
        "amount"           : txn.amount,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score"       : risk_score,
        "risk_level"       : risk_level,
        "device_type"      : txn.device_type,
        "location"         : txn.location,
        "tags"             : (["fraud_alert", "high_risk"]
                              if risk_level == "HIGH" else
                              ["fraud_alert", "medium_risk"]
                              if risk_level == "MEDIUM" else
                              ["normal"]),
    }
    _post_to_logstash(siem_event)

    # ── Twilio SMS for HIGH risk ────────────────────
    alert_sent = False
    if risk_level == "HIGH":
        _send_sms(txn_id, txn.amount, risk_score, txn.sender_upi)
        alert_sent = True

    # ── Persist to MongoDB / memory ────────────────
    record = {**siem_event, **txn.dict(), "alert_sent": alert_sent}
    _save_record(record)

    return {
        "transaction_id"   : txn_id,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score"       : risk_score,
        "risk_level"       : risk_level,
        "alert_sent"       : alert_sent,
        "boosts_applied"   : risk_result["boosts_applied"],
        "timestamp"        : timestamp,
    }

# ══════════════════════════════════════════════════════
# GET /transactions
# ══════════════════════════════════════════════════════
@app.get("/transactions")
async def get_transactions(limit: int = 50, risk_level: Optional[str] = None):
    records = _get_records(limit * 3)
    if risk_level:
        records = [r for r in records if r.get("risk_level") == risk_level.upper()]
    return {"transactions": records[:limit], "count": len(records[:limit])}

# ══════════════════════════════════════════════════════
# GET /analytics
# ══════════════════════════════════════════════════════
@app.get("/analytics")
async def get_analytics():
    records = _get_records(500)
    total   = len(records)
    if total == 0:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "avg_risk_score": 0}

    high   = sum(1 for r in records if r.get("risk_level") == "HIGH")
    medium = sum(1 for r in records if r.get("risk_level") == "MEDIUM")
    low    = sum(1 for r in records if r.get("risk_level") == "LOW")
    scores = [r["risk_score"] for r in records if "risk_score" in r]
    avg    = round(sum(scores) / len(scores), 2) if scores else 0

    return {
        "total"          : total,
        "high"           : high,
        "medium"         : medium,
        "low"            : low,
        "avg_risk_score" : avg,
        "high_pct"       : round(high / total * 100, 1),
        "medium_pct"     : round(medium / total * 100, 1),
        "low_pct"        : round(low / total * 100, 1),
        "recent_scores"  : [r.get("risk_score", 0) for r in records[:20]],
    }

# ══════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status"    : "ok",
        "models"    : MODELS_OK,
        "mongodb"   : MONGO_OK,
        "twilio"    : TWILIO_OK,
        "logstash"  : "http://localhost:5000",
        "timestamp" : datetime.utcnow().isoformat() + "Z",
    }

# ── Run directly ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
