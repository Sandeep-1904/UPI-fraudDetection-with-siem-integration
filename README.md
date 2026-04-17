# FraudShield – AI-Powered UPI Fraud Detection System with Real-Time SIEM Integration

FraudShield is a full-stack fraud detection system for UPI transactions. It uses machine learning to predict fraud probability, a hybrid risk scoring engine to classify transaction risk, and SIEM integration for real-time monitoring and investigation.

## Features

* Real-time UPI fraud detection
* Machine learning models:

  * Random Forest
  * Gradient Boosting
* Hybrid risk scoring engine (AI + rule-based boosts)
* FastAPI backend with REST endpoints
* ELK Stack SIEM integration:

  * Elasticsearch
  * Logstash
  * Kibana
* Twilio SMS alerts for high-risk transactions
* MongoDB transaction storage
* Analytics dashboard with charts and live transaction logs
* Graceful fallback if MongoDB, Twilio, or Logstash are unavailable

## Project Architecture

The system is designed in 5 layers:

1. **Data Layer**
   Generates and stores synthetic UPI transaction data.

2. **Machine Learning Layer**
   Prepares features, balances data using SMOTE, trains models, and saves trained artifacts.

3. **API Layer**
   FastAPI backend handles prediction, analytics, transaction retrieval, and health checks.

4. **SIEM Layer**
   Logs medium- and high-risk events into ELK Stack for centralized monitoring.

5. **Presentation Layer**
   Dashboard for user interaction and analytics visualization.

## Dataset

The project uses a synthetic dataset because real UPI fraud data is not publicly available.

* Total transactions: **5,500**
* Normal transactions: **5,000**
* Fraud transactions: **500**
* Fraud rate: **9.1%**

### Fraud patterns included

* Large-value suspicious transfers
* Micro-transaction probing
* Emulator or rooted devices
* Unknown or foreign locations
* VPN / TOR / masked IP addresses
* High transaction frequency
* New receivers
* Midnight to 5 AM fraud activity

## Machine Learning Pipeline

1. Load CSV dataset
2. Encode categorical features
3. Perform feature engineering
4. Split into training and testing sets
5. Apply SMOTE on training data
6. Train:

   * Random Forest
   * Gradient Boosting
7. Evaluate and save models

### Engineered Features

* `is_odd_hour`
* `is_large_amount`
* `is_high_frequency`
* `risk_device`
* `risk_location`

## Risk Scoring Engine

The final risk score is computed using:

`Risk Score = min(100, fraud_probability × 100 + rule_boosts)`

### Rule boosts

* `+15` suspicious device
* `+15` suspicious location
* `+10` amount > ₹50,000
* `+10` frequency > 10/hr
* `+8` transaction during 0–5 AM
* `+7` new receiver

### Risk levels

* **HIGH**: score ≥ 70
* **MEDIUM**: score 40–69
* **LOW**: score < 40

## API Endpoints

### `POST /predict`

Scores a transaction and returns:

* transaction ID
* fraud probability
* risk score
* risk level
* alert status
* applied boosts
* timestamp

### `GET /transactions`

Returns recent transactions with optional filtering.

### `GET /analytics`

Returns dashboard statistics:

* total transactions
* HIGH / MEDIUM / LOW counts
* average risk score
* recent scores

### `GET /health`

Checks status of:

* model loading
* MongoDB
* Twilio
* Logstash

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

### ML / Data

* scikit-learn
* imbalanced-learn
* pandas
* numpy
* Faker
* joblib

### Database / Alerts

* MongoDB
* PyMongo
* Twilio

### SIEM

* Elasticsearch
* Logstash
* Kibana
* Docker Compose

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

## Project Structure

```bash
upi_fraud_system/
├── generate_dataset.py
├── train_model.py
├── risk_engine.py
├── main.py
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── upi_transactions.csv
├── models/
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── le_location.pkl
│   ├── le_device.pkl
│   └── feature_cols.pkl
├── static/
│   └── dashboard.html
└── logstash/
    ├── pipeline/
    │   └── fraudshield.conf
    └── config/
        └── logstash.yml
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraudshield.git
cd fraudshield
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from `.env.example`.

Example:

```env
TWILIO_SID=your_twilio_sid
TWILIO_TOKEN=your_twilio_token
TWILIO_FROM=your_twilio_number
TWILIO_ADMIN=admin_number
MONGO_URI=mongodb://localhost:27017/
```

## Running the Project

### Step 1: Generate dataset

```bash
python generate_dataset.py
```

### Step 2: Train models

```bash
python train_model.py
```

### Step 3: Start ELK Stack

```bash
docker compose up -d
```

### Step 4: Run FastAPI server

```bash
uvicorn main:app --reload --port 8000
```

### Step 5: Open dashboard

* App: `http://localhost:8000`
* API Docs: `http://localhost:8000/docs`
* Kibana: `http://localhost:5601`

## Example Prediction Request

```json
{
  "sender_upi": "fraud@paytm",
  "receiver_upi": "merchant@ybl",
  "amount": 99000,
  "payment_type": "P2P",
  "hour_of_day": 2,
  "day_of_week": 1,
  "device_type": "Emulator",
  "location": "Unknown_IP",
  "is_new_receiver": 1,
  "txn_frequency_1hr": 15,
  "failed_pin_attempts": 3,
  "vpn_used": 1
}
```

## Example Response

```json
{
  "transaction_id": "TXN4A9F2B1C3D",
  "fraud_probability": 0.9821,
  "risk_score": 100.0,
  "risk_level": "HIGH",
  "alert_sent": true,
  "boosts_applied": [
    "risk_device +15",
    "risk_location +15",
    "odd_hour +8"
  ],
  "timestamp": "2026-04-15T10:20:30Z"
}
```

## Testing Highlights

* 16 test cases executed
* 16 passed
* 0 failed
* Average response time: ~18 ms
* Model AUC-ROC: 1.0 on synthetic test data

## Limitations

* Uses synthetic dataset, not real bank transaction data
* Perfect performance is due to clean synthetic fraud patterns
* Not deployed to cloud/production environment
* No live UPI gateway integration
* No authentication/authorization layer yet

## Future Enhancements

* Real-world dataset integration
* Kafka for streaming transactions
* Deep learning / autoencoder-based anomaly detection
* JWT authentication
* Cloud deployment
* Role-based analyst access
* Better explainability with SHAP/LIME

## Use Cases

* Fraud detection prototype
* Cybersecurity + ML portfolio project
* SIEM demonstration project
* Real-time analytics dashboard project

## Author

**Mulpuru Siva Sandeep**
B.Tech CSE

## License

This project is developed for academic and educational purposes.

---
