import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

try:
    # Load model
    model_path = "artifacts/model_trainer/model.pkl"
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Load data
    print("Loading data from artifacts/data_transformation/...")
    X_train = pd.read_csv("artifacts/data_transformation/X_train.csv")
    y_train = pd.read_csv("artifacts/data_transformation/y_train.csv").values.ravel()
    X_test = pd.read_csv("artifacts/data_transformation/X_test.csv")
    y_test = pd.read_csv("artifacts/data_transformation/y_test.csv").values.ravel()

    # Predict
    print("Predicting...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_acc}")
    print(f"Testing Accuracy: {test_acc}")

except Exception as e:
    print(f"Error: {e}")
