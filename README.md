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

---

## Instructions to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shubhangikad25-max/urbanflow-ai.git
   cd urbanflow-ai
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   - Ensure `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv` are in the `data/` folder
   - These files should already be included in the repository

5. **Train Models (Optional)**
   If you want to retrain the models:
   ```bash
   python run_final_models.py
   ```
   This will generate trained model files in the `models/` directory.

6. **Run the Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```
   The app will launch at: `http://localhost:8501`

7. **Access the Application**
   - Open your browser and navigate to `http://localhost:8501`
   - Use the sidebar to navigate between different analytics pages
   - Explore Executive Overview, Traffic Analytics, Geospatial Intelligence, and more

### Project Structure

```
├── dashboard/
│   ├── app.py              # Main Streamlit application
│   └── *.py                # Helper modules for dashboard
├── src/
│   ├── preprocessing.py    # Data preprocessing
│   ├── feature_engineering.py  # Feature extraction
│   ├── train_model.py      # Model training logic
│   ├── ensemble.py         # Ensemble model implementation
│   ├── inference.py        # Prediction inference
│   └── utils.py            # Utility functions
├── data/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
│   └── sample_submission.csv
├── models/
│   └── *.pkl               # Trained model files (generated after training)
├── notebooks/
│   └── traffic_prediction.ipynb  # Jupyter notebook with analysis
├── outputs/
│   └── *.csv               # Prediction outputs
├── requirements.txt        # Python dependencies
├── run_final_models.py     # Script to train all models
├── analyze_data.py         # Data analysis script
└── README.md               # Project documentation
```

### Features to Explore

1. **Executive Overview** - High-level metrics and KPIs
2. **Traffic Analytics** - Temporal and spatial traffic patterns
3. **Geospatial Intelligence** - Location-based demand analysis
4. **Weather Intelligence** - Weather impact on traffic
5. **AI Prediction Center** - Real-time traffic demand predictions
6. **Event Impact Simulation** - Test scenarios (concerts, festivals, sports)
7. **Emergency Simulation** - Incident impact analysis (accidents, road closures, severe weather)
8. **Smart City Recommendations** - AI-driven actionable insights for urban planners

### Troubleshooting

- **Port 8501 already in use?** Use: `streamlit run dashboard/app.py --server.port 8502`
- **Missing models?** Run `python run_final_models.py` to generate trained models
- **Data files not found?** Ensure CSV files are in the `data/` directory
- **Dependency errors?** Run `pip install --upgrade -r requirements.txt`

### Model Performance

- **Ensemble R² Score:** 0.884
- **Dataset:** 77,299 samples
- **Locations:** 1,249 unique geospatial zones
- **Algorithms:** CatBoost + LightGBM + XGBoost

---

**For questions or issues, please open an issue on GitHub.**