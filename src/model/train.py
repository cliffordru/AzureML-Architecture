"""
Sample training script demonstrating Azure ML training pipeline.
"""

import os
import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data():
    """
    Load and prepare dataset.
    Replace this with your actual data loading logic.
    """
    # Sample data loading (replace with your dataset)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    return pd.DataFrame(X), pd.Series(y)

def prepare_data(X, y):
    """Prepare data for training."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train the model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics."""
    y_pred = model.predict(X_test)
    
    # Log metrics with MLflow
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Generate and log detailed classification report
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, "classification_report.txt")
    
    return report

def main():
    # Start MLflow run
    with mlflow.start_run():
        # Load and prepare data
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluation_report = evaluate_model(model, X_test, y_test)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")
        
        print("Model training completed successfully!")
        print("\nEvaluation Report:")
        print(evaluation_report)

if __name__ == "__main__":
    main()
