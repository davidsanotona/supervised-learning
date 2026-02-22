import pandas as pd
from models.pricing_model import TaxiPricingModel

def run_nyc_pipeline():
    # 1. Load Data
    print("Loading NYC Taxi Sample Data...")
    df = pd.read_csv('data/ssample.csv')

    # 2. Initialize and Process
    pricing_engine = TaxiPricingModel()
    X, y = pricing_engine.preprocess_data(df)

    # 3. Train and Evaluate
    print("Training Linear Regression Model...")
    train_rmse = pricing_engine.train(X, y)

    print("\n" + "="*30)
    print("NYC TAXI PRICING METRICS")
    print("="*30)
    print(f"Training RMSE: {train_rmse:.2f}")
    print("="*30)

if __name__ == "__main__":
    run_nyc_pipeline()