Project Name:
UrbanFlow AI: AI-Powered Event-Driven Congestion Forecasting & Smart Mobility Intelligence Platform

Problem Statement:
Event-Driven Congestion (Planned & Unplanned) and Traffic Demand Prediction

Approach:
Developed a machine learning solution to predict traffic demand using location, weather, road infrastructure, and temporal features.
Features Event Impact Simulations (Concerts, Festivals, Sports) and Emergency Incident Simulations (Road closures, Accidents, Heavy rainfall) with Smart City Recommendations.

Feature Engineering:
- Hour extraction from timestamp
- Peak-hour identification
- Geohash encoding
- RoadType encoding
- Weather encoding
- Interaction features

Models Used:
- CatBoost Regressor
- LightGBM Regressor
- XGBoost Regressor
- Ensemble Learning

Evaluation Metric:
R² Score

Tools & Technologies:
- Python
- Pandas
- NumPy
- Scikit-Learn
- CatBoost
- LightGBM
- XGBoost
- Streamlit
- Plotly

Output:
Generated submission.csv with predictions for all test records.