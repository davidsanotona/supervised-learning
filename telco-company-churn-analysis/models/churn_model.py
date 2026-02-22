import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class ChurnRiskModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.feature_cols = []

    def preprocess_data(self, df):
        """
        Cleans raw Telco signals and prepares dummy variables.
        """
        # 1. Clean TotalCharges (Handle missing/empty strings)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
        
        # 2. Drop unique IDs not useful for modeling
        df = df.drop(columns=['customerID'], errors='ignore')
        
        # 3. Define target and features
        # Mapping target 'Churn' to binary
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # 4. One-Hot Encoding for categorical features (Contract, TechSupport, etc.)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.drop(columns=['Churn'])
        y = df_encoded['Churn']
        
        # Store metadata for Revenue-at-Risk calculation
        self.feature_cols = X.columns.tolist()
        
        return X, y, df['MonthlyCharges']

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def calculate_revenue_at_risk(self, X, monthly_charges):
        """
        Quantifies financial exposure based on churn probability.
        """
        probs = self.model.predict_proba(X)[:, 1]
        at_risk_df = pd.DataFrame({
            'Churn_Probability': probs,
            'Monthly_Charges': monthly_charges
        })
        # Calculate Expected Revenue Loss
        at_risk_df['Expected_Loss'] = at_risk_df['Churn_Probability'] * at_risk_df['Monthly_Charges']
        return at_risk_df['Expected_Loss'].sum()