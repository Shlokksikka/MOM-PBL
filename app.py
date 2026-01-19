
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="PBL: CSEB AI Designer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (The "Wow" Factor)
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("Sustainable Construction AI: CSEB Designer")
st.markdown("### End-to-End Machine Learning Pipeline for Green Building Materials")

# Sidebar Navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1599690940375-749e793bbce4?q=80&w=2000&auto=format&fit=crop", use_column_width=True)
    st.title("Project Navigator")
    options = st.radio("Go to", ["1. Project Synopsis", "2. Data Central", "3. Statistical Analysis 3D", "4. Model Arena", "5. Prediction Tool", "6. Smart Mix Optimizer"])
    st.info("PBL Project: Sustainable Materials")

# --- AUTO-SETUP LOGIC ---
# Automatically load data and train model if not present
if 'data' not in st.session_state:
    try:
        # Try loading default synthetic data
        st.session_state['data'] = pd.read_csv("cseb_dataset.csv")
    except:
        pass # Handle elegantly if file doesn't exist

if 'data' in st.session_state and 'trained_model' not in st.session_state:
    # Auto-train a default Random Forest model so users don't hit walls
    df = st.session_state['data']
    target = 'Compressive_Strength_MPa'
    if target in df.columns:
        features = df.columns.drop(target)
        X = df[features]
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save to session
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = features
        st.session_state['target_name'] = target
# ------------------------

# 1. Project Synopsis
if options == "1. Project Synopsis":
    st.header("Project Synopsis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("""
        **Topic**: AI-Driven Mix Design for Compressed Stabilised Earth Blocks (CSEB).
        
        **Problem**: Traditional soil testing takes **28 days**. This delays construction and leads to cement wastage.
        
        **Solution**: An intelligent "Virtual Lab" that predicts block strength instantly and **optimizes mix ratios** for sustainability.
        """)
        
        st.markdown("### Objectives")
        st.markdown("""
        - **Analyze**: Discover how soil grading impacts strength.
        - **Predict**: Train ML models (RF, ANN) to forecast performance.
        - **Optimize**: Find the "Greenest" mix (Minimizing Cement) that still meets safety standards.
        """)
    with col2:
        st.metric(label="Testing Time Saved", value="28 Days", delta="100%")
        st.metric(label="Potential CO2 Reduction", value="~15%", delta="Optimized Cement")

# 2. Data Central
elif options == "2. Data Central":
    st.header("Data Collection & Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Lab CSV", type=["csv"])
    
    with col2:
        st.markdown("#### Quick Load")
        if st.button("Load Synthetic Data (CSEB Default)"):
            try:
                df = pd.read_csv("cseb_dataset.csv")
                st.session_state['data'] = df
                st.success("Loaded Synthetic CSEB Data.")
            except:
                st.error("Generate data first!")
        
        if st.button("Load Real-World Concrete Data (UCI)"):
            from generate_data import get_real_data
            with st.spinner("Fetching from Repository..."):
                real_df = get_real_data()
                if real_df is not None:
                    st.session_state['data'] = real_df
                    st.success("Loaded Real Data.")

    if uploaded_file:
        st.session_state['data'] = pd.read_csv(uploaded_file)
            
    if 'data' in st.session_state:
        st.markdown("### Dataset Overview")
        st.dataframe(st.session_state['data'].head(), use_container_width=True)
        st.write(st.session_state['data'].describe())

# 3. Statistical Analysis
elif options == "3. Statistical Analysis 3D":
    st.header("Advanced Exploratory Analysis")
    
    if 'data' not in st.session_state:
        st.error("Please load data first.")
    else:
        df = st.session_state['data']
        target_col = 'Compressive_Strength_MPa'
        if target_col not in df.columns: target_col = df.columns[-1]

        # 3D Plot
        st.subheader("Interactive 3D Factor Analysis")
        st.markdown("Visualize the complex interaction between **Cement**, **Compaction/Water**, and **Strength**.")
        
        # Try to guess logical columns for 3D plot
        cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Attempt to pre-select relevant columns for the 3D plot
        default_x = 'Cement_Content_percent' if 'Cement_Content_percent' in cols else (cols[0] if len(cols) > 0 else None)
        default_y = 'Compaction_Pressure_MPa' if 'Compaction_Pressure_MPa' in cols else ('Water_Content_percent' if 'Water_Content_percent' in cols else (cols[1] if len(cols) > 1 else None))
        default_z = target_col
        
        x_axis = st.selectbox("X Axis (e.g. Cement)", cols, index=cols.index(default_x) if default_x in cols else 0)
        y_axis = st.selectbox("Y Axis (e.g. Pressure/Water)", cols, index=cols.index(default_y) if default_y in cols else (1 if len(cols) > 1 else 0))
        z_axis = st.selectbox("Z Axis (Target Strength)", cols, index=cols.index(default_z) if default_z in cols else (len(cols)-1 if len(cols) > 0 else 0))
        
        if x_axis and y_axis and z_axis:
            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                                color=z_axis, size_max=18, opacity=0.7,
                                color_continuous_scale='Viridis')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns in the dataset to create a 3D plot.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation Matrix")
            fig_corr = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr)
        with col2:
            st.subheader("Distribution Analysis")
            fig_dist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}", color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig_dist)

# 4. Model Arena
elif options == "4. Model Arena":
    st.header("Model Arena: Algorithm Comparison")
    
    if 'data' not in st.session_state:
        st.error("Load data first.")
    else:
        df = st.session_state['data']
        target = 'Compressive_Strength_MPa'
        if target not in df.columns: target = df.columns[-1]
        
        features = df.columns.drop(target)
        X = df[features]
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox("Choose Gladiator", ["Linear Regression", "Random Forest", "SVR", "Neural Network (ANN)"])
        
        if st.button("Train & Evaluate"):
            model = None
            if model_name == "Linear Regression": model = LinearRegression()
            elif model_name == "Random Forest": model = RandomForestRegressor(n_estimators=100)
            elif model_name == "SVR": model = SVR()
            elif model_name == "Neural Network (ANN)": model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000)
            
            with st.spinner("Training..."):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)
            
            st.success(f"**{model_name}** Results:")
            
            # Nice Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{r2:.3f}", delta="Higher is better")
            c2.metric("RMSE", f"{np.sqrt(mse):.3f}", delta="Lower is better", delta_color="inverse") # Changed MSE to RMSE
            c3.metric("Samples", f"{len(y_test)}")
            
            # Save for Prediction/Opt
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = features
            st.session_state['target_name'] = target
            
            # Plot
            fig_acc = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': "Predicted"}, title="Prediction Accuracy")
            fig_acc.add_shape(type="line", line=dict(dash="dash"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig_acc)

# 5. Prediction Tool
elif options == "5. Prediction Tool":
    st.header("Virtual Lab: Strength Predictor")
    
    if 'trained_model' not in st.session_state:
        st.warning("Train a model first!")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        feats = st.session_state['feature_names']
        
        input_data = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            with cols[i%3]:
                # Use mean of the feature from the original data as default value
                default_val = float(st.session_state['data'][f].mean())
                input_data[f] = st.number_input(f, value=default_val, format="%.2f")
        
        if st.button("Predict Strength"):
            val_df = pd.DataFrame([input_data])
            # Ensure columns are in the same order as during training
            val_df = val_df[feats] 
            val_scaled = scaler.transform(val_df)
            pred = model.predict(val_scaled)[0]
            
            st.balloons()
            st.metric(label="Predicted Compressive Strength", value=f"{pred:.2f} MPa")
            
            if pred > 7:
                 st.success("Meets Structural Standards (e.g., 7 MPa for load-bearing).")
            elif pred > 3.5:
                 st.info("Suitable for non-load-bearing or low-stress applications.")
            else:
                 st.warning("Below typical structural standards. Consider increasing stabilizer or compaction.")

# 6. Smart Mix Optimizer
elif options == "6. Smart Mix Optimizer":
    st.header("Eco-Smart Mix Optimizer")
    st.markdown("""
    **The Problem**: "I need a block with **7 MPa** strength. What is the *minimum* cement I can use to save money and CO2?"
    
    **The Solution**: This tool reverses the ML model to find your "Golden Recipe".
    """)
    
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model (preferably Random Forest or ANN) first!")
    else:
        target_strength = st.slider("Target Strength (MPa)", 2.0, 15.0, 7.0, 0.5)
        strength_tolerance = st.slider("Strength Tolerance (MPa)", 0.0, 1.0, 0.5, 0.1)
        
        # Identify the "Cost/Eco" column (usually Cement)
        feats = st.session_state['feature_names']
        cement_col_options = [c for c in feats if 'cement' in c.lower() or 'stabilizer' in c.lower() or 'binder' in c.lower()]
        
        if cement_col_options:
            cement_col = st.selectbox("Select feature to minimize (e.g., Cement, Stabilizer)", cement_col_options, index=0)
            st.info(f"Optimizing to minimize: **{cement_col}**")
        else:
             cement_col = feats[0] # Fallback to first feature
             st.warning(f"Could not auto-detect cement/stabilizer column. Minimizing: **{cement_col}**")

        if st.button("Run Optimizer"):
            # Monte Carlo Simulation to find best mix
            # We generate 5000 random valid mixes based on data ranges
            df_ref = st.session_state['data']
            mins = df_ref[feats].min()
            maxs = df_ref[feats].max()
            
            # Generate random samples
            n_sims = 10000 # Increased simulations for better coverage
            random_mixes = pd.DataFrame()
            for col in feats:
                random_mixes[col] = np.random.uniform(mins[col], maxs[col], n_sims)
            
            # Predict
            scaler = st.session_state['scaler']
            mixes_scaled = scaler.transform(random_mixes)
            preds = st.session_state['trained_model'].predict(mixes_scaled)
            random_mixes['Predicted_Strength'] = preds
            
            # Filter matches (Allows +/- tolerance)
            valid_mixes = random_mixes[
                (random_mixes['Predicted_Strength'] >= target_strength - strength_tolerance) &
                (random_mixes['Predicted_Strength'] <= target_strength + strength_tolerance)
            ]
            
            if valid_mixes.empty:
                st.error("No mix found that achieves this strength within the given tolerance and data ranges. Try adjusting target strength or tolerance.")
            else:
                # Find best (lowest cement)
                best_mix = valid_mixes.sort_values(by=cement_col).iloc[0]
                
                st.success(f"Found {len(valid_mixes)} valid formulas!")
                st.subheader("The Golden Recipe (Lowest Eco-Impact)")
                
                # Show results nicely
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.metric(f"Minimum {cement_col}", f"{best_mix[cement_col]:.2f}")
                    st.metric("Predicted Strength", f"{best_mix['Predicted_Strength']:.2f} MPa")
                
                with col_res2:
                    st.write("Full Mix Definition:")
                    # Display only the feature columns and predicted strength
                    display_mix = best_mix[feats.tolist() + ['Predicted_Strength']].to_frame().T
                    st.dataframe(display_mix.style.format("{:.2f}"), use_container_width=True)

