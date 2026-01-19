import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="CSEB Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

# Title and Header
st.title("🧱 Compressed Stabilised Earth Blocks (CSEB) Strength Initializer")
st.markdown("### Machine Learning Solutions for Sustainable Construction")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Project Info", "Data Collection", "Data Analysis (EDA)", "Model Training", "Prediction Tool"])

# 1. Project Info
if options == "Project Info":
    st.image("https://images.unsplash.com/photo-1599690940375-749e793bbce4?q=80&w=2000&auto=format&fit=crop", caption="Optimized Earth Blocks", use_column_width=True)
    st.header("Project Synopsis")
    st.info("""
    **Topic**: Development of Machine Learning Models for CSEB Compressive Strength Prediction.
    
    Compressed Stabilised Earth Blocks (CSEB) are sustainable alternatives to conventional masonry. 
    Their strength depends on soil gradation, stabilizer content, compaction pressure, and curing conditions.
    
    **Goal**: Develop an intelligent tool to estimate compressive strength, reducing the need for extensive laboratory trials.
    """)
    
    st.subheader("Objectives")
    st.markdown("""
    - **Collect data** on soil & process parameters.
    - **Preprocess data** (normalization, scaling).
    - **Build ML models**: Linear Regression, Random Forest, SVR, ANN.
    - **Identify influential variables**.
    - **Develop a prediction tool** for end-users.
    """)

# 2. Data Collection
elif options == "Data Collection":
    st.header("Data Collection & Preview")
    
    st.markdown("Load the dataset containing soil properties/mix design and the target strength.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload functionality
        uploaded_file = st.file_uploader("Option A: Upload Your CSV", type=["csv"])
    
    with col2:
        st.markdown("**Option B: Use Available Datasets**")
        if st.button("Load Synthetic CSEB Data"):
            try:
                # Fallback to local generated file if available
                df = pd.read_csv("cseb_dataset.csv")
                st.session_state['data'] = df
                st.success("Loaded Synthetic CSEB Data.")
            except FileNotFoundError:
                st.error("File not found. Run generate_data.py first.")
        
        if st.button("Load Real-World Concrete Data (UCI Proxy)"):
            from generate_data import get_real_data
            with st.spinner("Downloading from UCI Repository..."):
                real_df = get_real_data()
                if real_df is not None:
                    st.session_state['data'] = real_df
                    st.success("Loaded Real-World Data from UCI Repository.")
                else:
                    st.error("Failed to download data.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success("Custom dataset uploaded successfully!")
            
    if 'data' in st.session_state:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state['data'].head())
        
        st.subheader("Dataset Statistics")
        st.write(st.session_state['data'].describe())

# 3. Statistical Analysis & EDA
elif options == "Data Analysis (EDA)":
    st.header("Statistical Analysis & Methodology")
    st.markdown("""
    This section applies rigorous statistical methods to understand the data before modeling. 
    We look for **Linear Relationships (Correlation)**, **Data Distribution (Normality)**, and **Outliers**.
    """)
    
    if 'data' not in st.session_state:
        st.error("Please load data in the 'Data Collection' tab first.")
    else:
        df = st.session_state['data']
        
        # A. Descriptive Statistics
        st.subheader("1. Descriptive Statistics")
        st.markdown("**Why it matters:** deviations in Mean vs Median can indicate skewness. High Standard Deviation implies high variability.")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
        
        col1, col2 = st.columns(2)
        
        # B. Correlation Analysis
        with col1:
            st.subheader("2. Pearson Correlation Matrix")
            st.markdown("**Objective:** Identify which features (inputs) satisfy the linear assumption with Strength.")
            
            # Select numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
            st.pyplot(fig)
            st.info("Values closer to **1.0** or **-1.0** indicate strong predictors.")
            
        # C. Distribution Analysis
        with col2:
            st.subheader("3. Target Distribution Analysis")
            st.markdown("**Objective:** Check for Normality (Gaussian Distribution). ML models assume normally distributed errors.")
            
            target_col = 'Compressive_Strength_MPa'
            if target_col not in df.columns:
                target_col = df.columns[-1] # Fallback to last column
            
            fig2, ax2 = plt.subplots()
            sns.histplot(df[target_col], kde=True, color='purple', ax=ax2)
            ax2.set_title(f"Distribution of {target_col}")
            st.pyplot(fig2)
            
            skewness = df[target_col].skew()
            st.metric("Skewness", f"{skewness:.4f}", delta_color="inverse")
            if abs(skewness) > 1:
                st.warning("Data is highly skewed. Consider Log-Transformation.")
            else:
                st.success("Distribution is relatively normal.")

        # D. Bivariate Analysis
        st.subheader("4. Feature vs. Strength Interaction")
        feature_to_plot = st.selectbox("Select Feature to Analyze", df.columns[:-1])
        
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df, x=feature_to_plot, y=target_col, hue=target_col, palette="viridis", ax=ax3)
        ax3.set_title(f"{feature_to_plot} vs {target_col}")
        st.pyplot(fig3)

# 4. Model Training
elif options == "Model Training":
    st.header("Model Training & Evaluation")
    
    if 'data' not in st.session_state:
        st.error("Please load data first.")
    else:
        df = st.session_state['data']
        
        # Feature Selection
        target = 'Compressive_Strength_MPa'
        features = df.columns.drop(target)
        
        X = df[features]
        y = df[target]
        
        # Scaling (Mock-up for some models)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-Test Split
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        st.write(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")
        
        # Model Selection
        model_name = st.selectbox("Select Algorithm", ["Linear Regression", "Random Forest", "Support Vector Regression (SVR)", "Artificial Neural Network (ANN)"])
        
        train_btn = st.button("Train Model")
        
        if train_btn:
            model = None
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Support Vector Regression (SVR)":
                model = SVR(kernel='rbf')
            elif model_name == "Artificial Neural Network (ANN)":
                model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                
            # Train
            with st.spinner(f"Training {model_name}..."):
                model.fit(X_train, y_train)
                
            # Evaluate
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            st.success(f"{model_name} Trained Successfully!")
            
            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")
            
            # Save model to session for Prediction tab
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = features
            
            # Feature Importance (only for RF)
            if model_name == "Random Forest":
                st.subheader("Feature Importance")
                importances = model.feature_importances_
                fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                st.bar_chart(fi_df.set_index('Feature'))
                
            # Visualization of Pred vs Actual
            st.subheader("Prediction vs Actual")
            fig4, ax4 = plt.subplots()
            ax4.scatter(y_test, predictions, alpha=0.7)
            ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax4.set_xlabel("Actual")
            ax4.set_ylabel("Predicted")
            st.pyplot(fig4)

# 5. Prediction Tool
elif options == "Prediction Tool":
    st.header("Predict CSEB Compressive Strength")
    
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model in the 'Model Training' tab first.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        
        st.markdown("Enter the parameters to predict the block strength:")
        
        # Create input fields dynamically
        input_data = {}
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 2]:
                val = st.number_input(f"{feature}", value=0.0)
                input_data[feature] = val
                
        if st.button("Predict Strength"):
            # Prepare input
            input_df = pd.DataFrame([input_data])
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            pred = model.predict(input_scaled)[0]
            
            st.success(f"Predicted Compressive Strength: **{pred:.2f} MPa**")
            
            # Interpretation (Simple logic)
            if pred > 5:
                st.info("This block strength is likely suitable for load-bearing walls (Class 3.5+).")
            else:
                st.warning("Low strength. Consider increasing stabilizer or compaction.")

