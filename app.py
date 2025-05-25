import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import os

# Set page configuration
st.set_page_config(page_title="SpaceX Launch Analysis & Prediction", layout="wide")

# Title and description
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("""
This platform visualizes SpaceX launch data, analyzes factors influencing launch success, 
and predicts future launch outcomes using machine learning. Weather data is included to 
enhance predictions. Explore historical data, view trends, and make predictions!

**Data Source**: Local CSV dataset with SpaceX launch and weather information
""")

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    # Load CSV
    csv_file = "weather.csv"
    if not os.path.exists(csv_file):
        st.error(f"CSV file '{csv_file}' not found. Please ensure it is in the same directory as app.py.")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Rename columns to match app structure
    df = df.rename(columns={
        'mission_name': 'name',
        'launch_date': 'date_utc',
        'payload_mass': 'mass_kg',
        'site_name': 'launchpad_name',
        'Weather': 'weather',
        'Temperature': 'temperature',
        'Wind': 'wind'
    })
    
    # Convert data types
    df['date_utc'] = pd.to_datetime(df['date_utc'], errors='coerce')
    df['success'] = df['success'].astype(bool)
    df['mass_kg'] = df['mass_kg'].astype(float)
    df['temperature'] = df['temperature'].astype(float)
    df['wind'] = df['wind'].astype(float)
    
    # Handle missing values
    df['rocket_name'] = df['rocket_name'].fillna('Unknown')
    df['launchpad_name'] = df['launchpad_name'].fillna('Unknown')
    df['orbit'] = df['orbit'].fillna('Unknown')
    df['location'] = df['location'].fillna('Unknown')
    df['weather'] = df['weather'].fillna('Unknown')
    df['temperature'] = df['temperature'].fillna(df['temperature'].median())
    df['wind'] = df['wind'].fillna(df['wind'].median())
    
    # Add year for filtering
    df['year'] = df['date_utc'].dt.year
    
    # Add core_flight and fairings columns (set defaults as not provided)
    df['core_flight'] = df.get('core_flight', 1.0)  # Assume single use unless specified
    df['fairings.reused'] = df.get('fairings.reused', False)
    df['fairings.recovery_attempt'] = df.get('fairings.recovery_attempt', False)
    df['fairings.recovered'] = df.get('fairings.recovered', False)
    
    # Add window (set default as not in CSV)
    df['window'] = df.get('window', 1800.0)  # Default: 30 minutes
    
    # Add approximate coordinates based on location
    location_coords = {
        'Cape Canaveral': [28.5623, -80.5774],
        'Vandenberg Space Force Base': [34.6328, -120.6107],
        'Omelek Island': [9.0478, 167.7431]
    }
    df['latitude'] = df['location'].map(lambda x: location_coords.get(x, [np.nan, np.nan])[0])
    df['longitude'] = df['location'].map(lambda x: location_coords.get(x, [np.nan, np.nan])[1])
    
    return df

# Load data
df = load_and_preprocess_data()
if df is None:
    st.error("Data loading failed. Please check the CSV file.")
    st.stop()

# Validate required columns
required_cols = ['success', 'rocket_name', 'launchpad_name', 'mass_kg', 'window', 'core_flight', 'weather', 'temperature', 'wind']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")
year_options = ['All'] + sorted(df['year'].unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", year_options)
rocket_options = ['All'] + sorted(df['rocket_name'].unique().tolist())
selected_rocket = st.sidebar.selectbox("Select Rocket Type", rocket_options)
launchpad_options = ['All'] + sorted(df['launchpad_name'].unique().tolist())
selected_launchpad = st.sidebar.selectbox("Select Launch Site", launchpad_options)
weather_options = ['All'] + sorted(df['weather'].unique().tolist())
selected_weather = st.sidebar.selectbox("Select Weather Condition", weather_options)

# Filter DataFrame
filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
if selected_rocket != 'All':
    filtered_df = filtered_df[filtered_df['rocket_name'] == selected_rocket]
if selected_launchpad != 'All':
    filtered_df = filtered_df[filtered_df['launchpad_name'] == selected_launchpad]
if selected_weather != 'All':
    filtered_df = filtered_df[filtered_df['weather'] == selected_weather]

# Data Overview
st.header("📊 Historical Launch Data")
st.write(f"Showing {len(filtered_df)} launches")
if len(filtered_df) > 0:
    st.dataframe(filtered_df[[
        'name', 'date_utc', 'rocket_name', 'launchpad_name', 'success', 'mass_kg', 'weather', 'temperature', 'wind'
    ]].head(10))
else:
    st.warning("No launches match the selected filters.")

# EDA Section
st.header("📈 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Launch Success Distribution")
    if len(filtered_df) > 0 and 'success' in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='success', ax=ax)
        ax.set_title('Launch Success Distribution')
        st.pyplot(fig)
    else:
        st.warning("No data available for Success Distribution plot.")

with col2:
    st.subheader("Payload Mass vs. Success")
    if len(filtered_df) > 0 and 'success' in filtered_df.columns and 'mass_kg' in filtered_df.columns:
        if filtered_df['mass_kg'].notna().any() and filtered_df['success'].nunique() > 1 and filtered_df['mass_kg'].max() > 0:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_df, x='success', y='mass_kg', ax=ax)
            ax.set_title('Payload Mass vs. Launch Success')
            st.pyplot(fig)
        else:
            st.warning("Insufficient or invalid payload mass data for Payload Mass vs. Success plot.")
    else:
        st.warning("No data available for Payload Mass vs. Success plot.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Rocket Type vs. Success")
    if len(filtered_df) > 0 and 'rocket_name' in filtered_df.columns and 'success' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='rocket_name', hue='success', ax=ax)
        ax.set_title('Rocket Type vs. Launch Success')
        xticks = sorted(filtered_df['rocket_name'].unique())
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for Rocket Type vs. Success plot.")

with col4:
    st.subheader("Weather vs. Success")
    if len(filtered_df) > 0 and 'weather' in filtered_df.columns and 'success' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='weather', hue='success', ax=ax)
        ax.set_title('Weather Condition vs. Launch Success')
        xticks = sorted(filtered_df['weather'].unique())
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for Weather vs. Success plot.")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Launches per Year")
    if len(filtered_df) > 0 and 'year' in filtered_df.columns and 'success' in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='year', hue='success', ax=ax)
        ax.set_title('Launches per Year by Success')
        xticks = sorted(filtered_df['year'].unique())
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for Launches per Year plot.")

with col6:
    st.subheader("Temperature vs. Success")
    if len(filtered_df) > 0 and 'temperature' in filtered_df.columns and 'success' in filtered_df.columns:
        if filtered_df['temperature'].notna().any() and filtered_df['success'].nunique() > 1:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_df, x='success', y='temperature', ax=ax)
            ax.set_title('Temperature vs. Launch Success')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for Temperature vs. Success plot.")
    else:
        st.warning("No data available for Temperature vs. Success plot.")

# Geospatial Visualization
st.header("🗺️ Launch Sites Map")
if len(filtered_df) > 0 and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()] if filtered_df['latitude'].notna().any() else [0, 0]
    m = folium.Map(location=map_center, zoom_start=3)
    for _, row in filtered_df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            color = 'green' if row['success'] else 'red'
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"{row['name']} ({row['launchpad_name']}) - {'Success' if row['success'] else 'Failure'}",
                icon=folium.Icon(color=color)
            ).add_to(m)
    st_folium(m, width=700, height=500)
else:
    st.warning("No valid location data available for the map.")

# Machine Learning Prediction
st.header("🔮 Predict Launch Success")
@st.cache_resource
def train_ml_model():
    ml_df = df[[
        'success', 'rocket_name', 'launchpad_name', 'mass_kg', 'window', 'core_flight',
        'fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered',
        'weather', 'temperature', 'wind'
    ]].copy()
    ml_df['success'] = ml_df['success'].astype(int)
    
    num_cols = ['mass_kg', 'window', 'core_flight', 'temperature', 'wind']
    valid_num_cols = [col for col in num_cols if ml_df[col].notna().any() and ml_df[col].max() > 0]
    imputer = SimpleImputer(strategy='median')
    if valid_num_cols:
        ml_df[valid_num_cols] = imputer.fit_transform(ml_df[valid_num_cols])
    else:
        st.warning("No valid numeric columns for imputation. Skipping imputation.")
    
    cat_cols = ['rocket_name', 'launchpad_name', 'weather']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        label_encoders[col] = le
    
    for col in ['fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered']:
        ml_df[col] = ml_df[col].astype(int)
    
    X = ml_df.drop(columns=['success'])
    y = ml_df['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    st.write(f"**Model Performance**: Random Forest CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    return rf_model, label_encoders, imputer, valid_num_cols

# Train model and get valid numeric columns
model, label_encoders, imputer, valid_num_cols = train_ml_model()

# Prediction input form
st.subheader("Input Launch Parameters")
with st.form("prediction_form"):
    rocket_name = st.selectbox("Rocket Type", sorted(df['rocket_name'].unique()))
    launchpad_name = st.selectbox("Launch Site", sorted(df['launchpad_name'].unique()))
    
    # Payload mass slider
    mass_kg = st.slider(
        "Payload Mass (kg)",
        min_value=0.0,
        max_value=float(df['mass_kg'].max()),
        value=float(df['mass_kg'].mean()),
        step=100.0
    )
    
    # Launch window slider
    window = st.slider(
        "Launch Window (seconds)",
        min_value=0.0,
        max_value=3600.0,  # Reasonable max: 1 hour
        value=1800.0,      # Default: 30 minutes
        step=60.0
    )
    
    core_flight = st.number_input("Core Flight Number", min_value=0, value=1)
    
    # Weather condition
    weather = st.selectbox("Weather Condition", sorted(df['weather'].unique()))
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (°C)",
        min_value=float(df['temperature'].min()),
        max_value=float(df['temperature'].max()),
        value=float(df['temperature'].mean()),
        step=1.0
    )
    
    # Wind speed slider
    wind = st.slider(
        "Wind Speed (km/h)",
        min_value=float(df['wind'].min()),
        max_value=float(df['wind'].max()),
        value=float(df['wind'].mean()),
        step=1.0
    )
    
    fairings_reused = st.checkbox("Fairings Reused")
    fairings_recovery_attempt = st.checkbox("Fairings Recovery Attempt")
    fairings_recovered = st.checkbox("Fairings Recovered")
    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = pd.DataFrame({
        'rocket_name': [rocket_name],
        'launchpad_name': [launchpad_name],
        'mass_kg': [mass_kg],
        'window': [window],
        'core_flight': [core_flight],
        'fairings.reused': [fairings_reused],
        'fairings.recovery_attempt': [fairings_recovery_attempt],
        'fairings.recovered': [fairings_recovered],
        'weather': [weather],
        'temperature': [temperature],
        'wind': [wind]
    })
    
    for col in ['rocket_name', 'launchpad_name', 'weather']:
        input_data[col] = label_encoders[col].transform(input_data[col])
    
    if valid_num_cols:
        input_data[valid_num_cols] = imputer.transform(input_data[valid_num_cols])
    
    for col in ['fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered']:
        input_data[col] = input_data[col].astype(int)
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    
    st.success(f"**Prediction**: {'Success' if prediction[0] == 1 else 'Failure'}")
    st.write(f"**Success Probability**: {probability[1]:.2%}")
    st.write(f"**Failure Probability**: {probability[0]:.2%}")

# Footer
st.markdown("""
---
**Developed by**: Kunal  
**Data Source**: SpaceX launch data with weather information (CSV)  
**Deployment**: Deployed on Cloud Streamlit.
""")
