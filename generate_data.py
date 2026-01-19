import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=100):
    np.random.seed(42)
    
    # Generate synthetic features
    # Soil composition (should sum to ~100%, simplifying for simulation)
    sand = np.random.uniform(40, 80, n_samples)
    clay = np.random.uniform(5, 30, n_samples)
    silt = 100 - sand - clay
    # Ensure no negative values (simple clamp)
    silt = np.maximum(silt, 0)
    
    # Recalculate to ensure sum is 100
    total = sand + clay + silt
    sand = (sand / total) * 100
    clay = (clay / total) * 100
    silt = (silt / total) * 100
    
    # Atterberg Limits
    liquid_limit = np.random.uniform(20, 50, n_samples)
    plastic_limit = np.random.uniform(10, 30, n_samples)
    // Plasticity Index cannot be negative
    plasticity_index = np.maximum(liquid_limit - plastic_limit, 0)
    
    # Mix parameters
    stabilizer_content = np.random.uniform(3, 12, n_samples) # % (e.g., 5%, 8%)
    compaction_pressure = np.random.uniform(2, 10, n_samples) # MPa
    moisture_content = np.random.uniform(8, 16, n_samples) # %
    
    # Simulate Compressive Strength
    # Formula (totally hypothetical linear combination + non-linearities + noise)
    # Strength increases with stabilizer, compaction pressure.
    # Strength has an optimal moisture content (parabolic).
    # Strength depends on soil suitability (less clay might be better for cement stabilization?).
    
    # Base strength
    strength = (
        0.5 * stabilizer_content + 
        0.3 * compaction_pressure + 
        0.05 * sand - 
        0.02 * clay - 
        0.1 * (moisture_content - 12)**2 # Optimal moisture around 12
    )
    
    # Add some randomness
    strength += np.random.normal(0, 1.5, n_samples)
    
    # Ensure positive strength
    strength = np.maximum(strength, 1.0)
    
    data = pd.DataFrame({
        'Sand_Content_%': sand,
        'Silt_Content_%': silt,
        'Clay_Content_%': clay,
        'Liquid_Limit_%': liquid_limit,
        'Plastic_Limit_%': plastic_limit,
        'Plasticity_Index_%': plasticity_index,
        'Stabilizer_Content_%': stabilizer_content,
        'Compaction_Pressure_MPa': compaction_pressure,
        'Moisture_Content_%': moisture_content,
        'Compressive_Strength_MPa': strength
    })
    
    return data

def get_real_data():
    """
    Downloads the 'Concrete Compressive Strength' dataset from a public repository.
    This serves as a real-world proxy for the CSEB dataset.
    """
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
    try:
        print(f"Downloading data from {url}...")
        df = pd.read_csv(url)
        # Rename columns to match our expected format somewhat, or keep as is and let the app handle it
        # The UCI dataset has columns like: cement, slag, ash, water, superplastic, coarseagg, fineagg, age, strength
        # We will map them to friendly names
        df.columns = [
            'Cement_kg_m3', 'Blast_Furnace_Slag_kg_m3', 'Fly_Ash_kg_m3', 
            'Water_kg_m3', 'Superplasticizer_kg_m3', 'Coarse_Aggregate_kg_m3', 
            'Fine_Aggregate_kg_m3', 'Age_Days', 'Compressive_Strength_MPa'
        ]
        return df
    except Exception as e:
        print(f"Error downloading real data: {e}")
        return None

if __name__ == "__main__":
    df = generate_synthetic_data(150)
    df.to_csv('cseb_dataset.csv', index=False)
    print("Synthetic dataset 'cseb_dataset.csv' created with 150 samples.")
