# AI-Driven Personal Financial Risk & Spending Behavior Analyzer

## 🚀 Overview

This project is an end-to-end **MLOps-powered financial risk prediction system** that analyzes user transaction behavior and predicts financial risk levels (Low / Medium / High).
It includes a complete machine learning pipeline, experiment tracking with MLflow, REST API deployment using FastAPI, and containerization with Docker.

---

## 🧠 Problem Statement

Individuals often struggle to understand whether their spending behavior is financially risky.
Traditional budgeting tools only show history but do not **predict future risk**.

This system uses machine learning to:

* Analyze income vs expenses
* Detect unhealthy spending behavior
* Predict financial risk probability
* Classify users into Low / Medium / High risk categories

---

## 🏗 Architecture

```
Data → Ingestion → Validation → Transformation → Model Training → Evaluation
                                         ↓
                                   MLflow Tracking
                                         ↓
                                   FastAPI Inference
                                         ↓
                                    Docker Container
```

---

## 📁 Project Structure

```
AI-Driven-Personal-Financial-Risk-Spending-Behavior-Analyzer/
│
├── app.py                      # FastAPI application
├── main.py                     # Pipeline runner
├── test_prediction.py          # Test inference script
├── requirements.txt
├── Dockerfile
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   └── model_trainer/
│       ├── model.pkl
│       └── feature_names.json
│
├── config/
│   ├── config.yaml
│   ├── params.yaml
│   └── schema.yaml
│
├── src/MLOPs/
│   ├── components/
│   ├── pipeline/
│   ├── entity/
│   ├── utils/
│   └── constants/
```

---

## 📊 Dataset

The dataset contains engineered behavioral features:

```
["Total_Income", "Total_Expense", "Num_Transactions", "Avg_Expense",
 "Max_Expense", "Low_Balance_Freq", "Expense_Income_Ratio", "Top_Category_Spend"]
```

---

## ⚙ ML Pipeline

### 1️⃣ Data Ingestion

* Loads local CSV file
* Stores in artifacts folder

### 2️⃣ Data Validation

* Schema check
* Column consistency validation

### 3️⃣ Data Transformation

* Scaling
* Train-test split

### 4️⃣ Model Training

* Logistic Regression classifier
* Metrics logged to MLflow

### 5️⃣ Risk Classification

```
Probability < 0.4 → LOW
0.4 – 0.7 → MEDIUM
> 0.7 → HIGH
```

---

## 📈 MLflow Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

Tracks:

* Parameters
* Metrics
* Model versions

---

## 🌐 FastAPI Service

Start API locally:

```bash
uvicorn app:app --reload
```

Open Swagger:

```
http://127.0.0.1:8000/docs
```

### Sample Request

```json
{
  "Total_Income": 60000,
  "Total_Expense": 42000,
  "Num_Transactions": 120,
  "Avg_Expense": 350,
  "Max_Expense": 2500,
  "Low_Balance_Freq": 3,
  "Expense_Income_Ratio": 0.7,
  "Top_Category_Spend": 15000
}
```

### Sample Response

```json
{
  "risk_probability": 0.23,
  "risk_level": "LOW"
}
```

---

## 🐳 Docker Deployment

Build:

```bash
docker build -t financial-risk-api .
```

Run:

```bash
docker run -p 8000:8000 financial-risk-api
```

Test:

```
http://127.0.0.1:8000/docs
```

---

## 🎯 Resume Value

* End-to-end MLOps pipeline
* MLflow experiment tracking
* REST API for predictions
* Dockerized deployment
* Feature schema consistency
* Real-world fintech problem

---

## 🔮 Future Enhancements

* User dashboard (Streamlit / React)
* Real-time transaction ingestion
* Model monitoring & drift detection
* Cloud deployment (AWS / GCP)

---

## 👤 Author

**Kavya Chougule**
AI / Data Engineering Enthusiast
