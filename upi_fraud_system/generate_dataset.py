"""
generate_dataset.py
===================
Produces upi_transactions.csv with:
  5,000 normal  transactions
    ₹10–₹10,000 | trusted devices | known cities | 6 AM–11 PM
  500   fraud   transactions
    >₹50,000 or <₹10 | Unknown/Foreign IP | Emulator/Unknown_Device
    new receiver | frequency >10/hr | midnight–5 AM
All rows shuffled before export.
"""

import random, hashlib, uuid
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED); np.random.seed(SEED)
fake = Faker("en_IN"); Faker.seed(SEED)

N_NORMAL, N_FRAUD = 5000, 500

UPI_HANDLES = ["@okaxis","@oksbi","@okicici","@ybl","@paytm","@ibl","@upi"]
KNOWN_CITIES = [
    "Mumbai","Delhi","Bangalore","Chennai","Hyderabad",
    "Kolkata","Pune","Ahmedabad","Jaipur","Lucknow",
    "Surat","Kanpur","Nagpur","Indore","Bhopal",
]
TRUSTED_DEVICES = [
    "Samsung_Galaxy_S23","OnePlus_11","Xiaomi_13",
    "iPhone_14","iPhone_15","Realme_10",
    "Vivo_V27","Oppo_Reno8","Google_Pixel_7","Motorola_Edge_40",
]
FRAUD_DEVICES = ["Emulator","Unknown_Device","Rooted_Android","VM_Instance"]
FRAUD_LOCATIONS = ["Unknown_IP","Foreign_IP","TOR_Exit_Node","VPN_IP"]
PAYMENT_TYPES = ["P2P","P2M","Recharge","Bill","QR"]

START = datetime(2025, 10, 1)
DAYS  = (datetime(2026, 3, 15) - START).days

def rand_ts(hour_min, hour_max):
    base = START + timedelta(days=random.randint(0, DAYS))
    return base.replace(
        hour=random.randint(hour_min, hour_max),
        minute=random.randint(0,59), second=random.randint(0,59)
    )

def upi(name):
    return name.lower().replace(" ","").replace(".","")[:8] + random.choice(UPI_HANDLES)

def dev_id(d):
    return hashlib.md5((d+str(random.randint(1000,9999))).encode()).hexdigest()[:14].upper()

def indian_ip():
    return f"49.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"

# ── NORMAL ─────────────────────────────────────────────
def normal():
    device = random.choice(TRUSTED_DEVICES)
    ts     = rand_ts(6, 23)
    return dict(
        txn_id              = "TXN" + uuid.uuid4().hex[:10].upper(),
        timestamp           = ts.strftime("%Y-%m-%d %H:%M:%S"),
        sender_upi          = upi(fake.name()),
        receiver_upi        = upi(fake.name()),
        amount              = round(random.uniform(10, 10_000), 2),
        payment_type        = random.choice(PAYMENT_TYPES),
        hour_of_day         = ts.hour,
        day_of_week         = ts.weekday(),
        device_type         = device,
        device_id           = dev_id(device),
        location            = random.choice(KNOWN_CITIES),
        ip_address          = indian_ip(),
        is_new_receiver     = random.choice([0,0,0,1]),
        txn_frequency_1hr   = random.randint(1, 4),
        failed_pin_attempts = random.choices([0,1],[90,10])[0],
        vpn_used            = 0,
        is_fraud            = 0,
    )

# ── FRAUD ──────────────────────────────────────────────
def fraud():
    device = random.choice(FRAUD_DEVICES)
    ts     = rand_ts(0, 5)
    amount = (round(random.uniform(50_001, 2_00_000), 2)
              if random.random() < 0.70 else
              round(random.uniform(0.50, 9.99), 2))
    return dict(
        txn_id              = "TXN" + uuid.uuid4().hex[:10].upper(),
        timestamp           = ts.strftime("%Y-%m-%d %H:%M:%S"),
        sender_upi          = upi(fake.name()),
        receiver_upi        = upi(fake.name()),
        amount              = amount,
        payment_type        = random.choice(PAYMENT_TYPES),
        hour_of_day         = ts.hour,
        day_of_week         = ts.weekday(),
        device_type         = device,
        device_id           = dev_id(device),
        location            = random.choice(["Unknown_City","Foreign_City","Spoofed_Location"]),
        ip_address          = random.choice(FRAUD_LOCATIONS),
        is_new_receiver     = 1,
        txn_frequency_1hr   = random.randint(11, 30),
        failed_pin_attempts = random.choices([0,1,2,3,4],[10,15,30,25,20])[0],
        vpn_used            = random.choice([0,1]),
        is_fraud            = 1,
    )

# ── BUILD & EXPORT ─────────────────────────────────────
print("Generating 5,000 normal transactions...")
rows = [normal() for _ in range(N_NORMAL)]
print("Generating 500 fraud transactions...")
rows += [fraud() for _ in range(N_FRAUD)]
random.shuffle(rows)

COLS = ["txn_id","timestamp","sender_upi","receiver_upi","amount","payment_type",
        "hour_of_day","day_of_week","device_type","device_id","location","ip_address",
        "is_new_receiver","txn_frequency_1hr","failed_pin_attempts","vpn_used","is_fraud"]

df = pd.DataFrame(rows)[COLS]
df.to_csv("upi_transactions.csv", index=False)

print(f"\n{'='*48}")
print("  upi_transactions.csv — Summary")
print(f"{'='*48}")
print(f"  Total   : {len(df):,}")
print(f"  Normal  : {(df.is_fraud==0).sum():,}")
print(f"  Fraud   : {(df.is_fraud==1).sum():,}  ({df.is_fraud.mean()*100:.1f}% fraud rate)")
print(f"  Columns : {len(df.columns)}")
print(f"\n  Saved → upi_transactions.csv")
