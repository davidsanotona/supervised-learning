# NYC Taxi Price Prediction Engine
Predictive Modeling for Dynamic Pricing Foundation


## Problem Statement
For urban mobility providers, pricing accuracy is critical for maintaining supply-demand equilibrium. This project addresses the challenge of predicting taxi fares in New York City by leveraging temporal, geographic, and historical signals. The goal is to build a robust Linear Regression model that estimates fares with high precision, allowing for better revenue forecasting and customer transparency.

## Technical Approach
- Feature Engineering: Extracted pick-up/drop-off coordinates, passenger counts, and temporal features (hour, day of week).
- Pipeline Modularization: Separated data preprocessing, feature scaling, and model training into a reusable class structure.
- Evaluation: Performance measured using Root Mean Squared Error (RMSE) to penalize large pricing deviations.