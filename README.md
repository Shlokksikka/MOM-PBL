# CSEB Strength Prediction App

This Streamlit application is designed for the PBL topic: "Development of Machine Learning Models for Compressed Stabilised Earth Blocks (CSEB) Strength Prediction".

## Features
- **Data Collection**: Upload existing lab data or use the built-in synthetic data generator.
- **Analysis**: Visualize correlations between soil properties/stabilizers and block strength.
- **Modeling**: Train and compare 4 different ML algorithms (Linear Regression, Random Forest, SVR, ANN).
- **Prediction**: Interactive tool to estimate strength for new mix designs.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data** (Optional, if you don't have a CSV):
   ```bash
   python generate_data.py
   ```
   This will create `cseb_dataset.csv`.

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Project Objectives Covered
- [x] Collection of literature/data (Simulated via script)
- [x] Preprocessing & Correlation Analysis
- [x] Building & Comparing ML Models
- [x] Feature Importance Analysis
- [x] Prediction Interface
