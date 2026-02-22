import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class TaxiPricingModel:
    def __init__(self):
        self.model = LinearRegression()
        # The final engineered features we will use for the model
        self.feature_cols = [] 

    def preprocess_data(self, df):
        """
        Translates raw NYC Taxi signals into a structured pricing pipeline.
        Includes temporal extraction and airport trip inference.
        """
        # 1. Convert timestamp to datetime objects
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        # 2. Infer Temporal Features from Pickup Time
        df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
        df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
        df['month'] = df['tpep_pickup_datetime'].dt.month
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 3. Infer Airport Trip from Airport_fee
        # Logic: If Airport_fee > 0, it's an airport trip
        df['is_airport_trip'] = df['Airport_fee'].apply(lambda x: 1 if x > 0 else 0)
        
        # 4. Binary encoding for store_and_fwd_flag
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).fillna(0)
        
        # 5. Define Categorical vs Numeric for One-Hot Encoding
        categorical_vars = ['VendorID', 'RatecodeID', 'day_of_week', 'hour_of_day', 'month']
        numeric_vars = ['passenger_count', 'is_airport_trip', 'is_weekend']
        
        # 6. Generate Dummy Variables (One-Hot Encoding)
        df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
        
        # 7. Final Feature Selection
        # Gather all dummy columns and numeric columns
        self.feature_cols = [col for col in df_encoded.columns if any(cv in col for cv in categorical_vars)]
        self.feature_cols += numeric_vars
        
        X = df_encoded[self.feature_cols]
        y = df_encoded['fare_amount']
        
        return X, y

    def train(self, X, y):
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        return rmse