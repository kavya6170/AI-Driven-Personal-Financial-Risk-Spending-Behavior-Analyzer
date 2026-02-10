💰 AI-Driven Personal Financial Risk & Spending Behavior Analyzer

An End-to-End MLOps Financial Intelligence System that analyzes personal transaction behavior and predicts financial risk levels (Low / Medium / High) using Machine Learning, MLflow experiment tracking, FastAPI deployment, Docker containerization, and CI/CD-ready architecture.

🚀 Project Highlights

End-to-End MLOps pipeline (Data → Model → Deployment)

Financial risk prediction using ML classification

Experiment tracking using MLflow

Real-time prediction API using FastAPI

Dockerized deployment for scalable production usage

Config-driven modular pipeline architecture

CI/CD-ready repository structure

🧠 Problem Statement

Traditional budgeting tools only provide historical insights and fail to predict financial risk behavior.

This system uses Machine Learning to:

Analyze spending patterns vs income

Detect risky financial behavior

Predict financial risk probability

Classify users into Low / Medium / High Risk

🏗️ System Architecture
Transaction Data
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Training & Evaluation
      ↓
MLflow Experiment Tracking
      ↓
FastAPI Prediction Service
      ↓
Docker Deployment

📂 Project Structure
AI-Driven-Personal-Financial-Risk-Spending-Behavior-Analyzer
│
├── app.py                # FastAPI service
├── main.py               # Pipeline execution
├── Dockerfile
├── requirements.txt
│
├── artifacts/            # Generated pipeline outputs
├── config/               # Config, params, schema
├── src/MLOPs/            # Modular pipeline source code
├── Dataset/              # Input dataset
└── templates/

📊 Dataset Features

The system predicts financial risk using behavioral features:

Total Income

Total Expense

Number of Transactions

Average Expense

Maximum Expense

Low Balance Frequency

Expense-Income Ratio

Top Category Spending

⚙️ ML Pipeline Stages
1️⃣ Data Ingestion

Loads dataset

Stores raw data in artifacts

2️⃣ Data Validation

Schema validation

Column consistency checks

3️⃣ Data Transformation

Feature scaling

Train-test split

4️⃣ Model Training

Logistic Regression classifier

Metrics logged to MLflow

5️⃣ Risk Classification
Probability	Risk Level
< 0.40	LOW
0.40 – 0.70	MEDIUM
> 0.70	HIGH
📈 MLflow Experiment Tracking

Run MLflow UI:

mlflow ui


Open:

http://127.0.0.1:5000


Tracks:

Parameters

Metrics

Model versions

🌐 Run FastAPI Service

Start API locally:

uvicorn app:app --reload


Swagger UI:

http://127.0.0.1:8000/docs

🐳 Docker Deployment

Build image:

docker build -t financial-risk-api .


Run container:

docker run -p 8000:8000 financial-risk-api

💼 Resume / Portfolio Value

This project demonstrates:

Real-world FinTech ML application

End-to-End MLOps engineering

Experiment tracking using MLflow

Production-ready API deployment

Docker-based scalable system design

Config-driven modular architecture

🔮 Future Enhancements

Streamlit / React financial dashboard

Real-time transaction ingestion pipeline

Model drift detection & monitoring

Cloud deployment (AWS / GCP)

User financial recommendation engine

👩‍💻 Author

Kavya Chougule
AI • Data Engineering • MLOps Enthusiast

⭐ Support

If you find this project useful, consider starring the repository to support the work.

📜 License

MIT License
