"""
risk_engine.py
==============
Risk Scoring Engine
  Score = min(100, fraud_probability × 100 + rule_boosts)

Rule boosts:
  Unknown/Emulator device   +15
  Foreign/Unknown location  +15
  Amount > ₹50,000          +10
  Frequency > 10/day        +10
  Hour 0–5 AM               +8
  New receiver              +7

Thresholds:
  HIGH   ≥ 70
  MEDIUM 40–69
  LOW    < 40
"""

def compute_risk_score(fraud_probability: float, txn: dict) -> dict:
    """
    Parameters
    ----------
    fraud_probability : float  (0.0 – 1.0 from model)
    txn               : dict   raw transaction fields

    Returns
    -------
    dict with keys: risk_score, risk_level, boosts_applied
    """
    base   = fraud_probability * 100
    boosts = []

    device   = str(txn.get("device_type", "")).lower()
    location = str(txn.get("location",    "")).lower()
    amount   = float(txn.get("amount",          0))
    freq     = float(txn.get("txn_frequency_1hr", 0))
    hour     = int(txn.get("hour_of_day",        12))
    new_recv = int(txn.get("is_new_receiver",    0))

    if any(k in device for k in ["unknown", "emulator", "rooted", "vm"]):
        base += 15
        boosts.append("risk_device +15")

    if any(k in location for k in ["unknown", "foreign", "spoofed", "tor", "vpn"]):
        base += 15
        boosts.append("risk_location +15")

    if amount > 50_000:
        base += 10
        boosts.append("large_amount +10")

    if freq > 10:
        base += 10
        boosts.append("high_frequency +10")

    if hour <= 5:
        base += 8
        boosts.append("odd_hour +8")

    if new_recv:
        base += 7
        boosts.append("new_receiver +7")

    score = min(100, round(base, 1))

    if score >= 70:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "risk_score"    : score,
        "risk_level"    : level,
        "boosts_applied": boosts,
    }


# ── Quick self-test ────────────────────────────────────
if __name__ == "__main__":
    print("Risk Engine — self test")
    print("-" * 40)

    high_risk = dict(device_type="Emulator", location="Unknown_IP",
                     amount=99000, txn_frequency_1hr=15,
                     hour_of_day=2, is_new_receiver=1)
    r = compute_risk_score(0.85, high_risk)
    print(f"HIGH  test → score={r['risk_score']}  level={r['risk_level']}")
    print(f"  boosts: {r['boosts_applied']}")
    assert r["risk_level"] == "HIGH" and r["risk_score"] >= 70, "FAIL"
    print("  PASS ✓")

    low_risk = dict(device_type="iPhone_14", location="Mumbai",
                    amount=500, txn_frequency_1hr=2,
                    hour_of_day=14, is_new_receiver=0)
    r = compute_risk_score(0.05, low_risk)
    print(f"\nLOW   test → score={r['risk_score']}  level={r['risk_level']}")
    assert r["risk_level"] == "LOW" and r["risk_score"] < 40, "FAIL"
    print("  PASS ✓")
