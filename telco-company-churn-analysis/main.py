import pandas as pd
from models.churn_model import ChurnRiskModel

def run_retention_pipeline():
    # 1. Load Data
    df = pd.read_csv('data/Telco-Customer-Churn.csv')

    # 2. Initialize
    risk_engine = ChurnRiskModel()
    X, y, monthly_charges = risk_engine.preprocess_data(df)

    # 3. Train
    accuracy = risk_engine.train(X, y)
    
    # 4. Quantify Risk
    total_risk = risk_engine.calculate_revenue_at_risk(X, monthly_charges)

    print("\n" + "="*40)
    print("TELCO RETENTION & RISK ANALYSIS")
    print("="*40)
    print(f"Model Accuracy: {accuracy:.2%}")
    print(f"Total Monthly Revenue at Risk: ${total_risk:,.2f}")
    print("="*40)

if __name__ == "__main__":
    run_retention_pipeline()