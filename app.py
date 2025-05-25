import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set page configuration
st.set_page_config(page_title="SpaceX Launch Analysis & Prediction", layout="wide")

# Title and description
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")
st.markdown("""
This platform visualizes SpaceX launch data, analyzes factors influencing launch success, 
and predicts future launch outcomes using machine learning. Explore historical data, 
view trends, and make predictions!

**Data Source**: [SpaceX API](https://api.spacexdata.com/v4/launches)
""")

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    # Define get_lookup_dict function
    def get_lookup_dict(url, id_key='id', value_key='name'):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                st.error(f"Response from {url} is not a list")
                return {}
            return {item[id_key]: item[value_key] for item in items if id_key in item and value_key in item}
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching {url}: {e}")
            return {}

    # Load or fetch data
    try:
        df = pd.read_csv("spacex_launches.csv")
    except FileNotFoundError:
        response = requests.get("https://api.spacexdata.com/v4/launches")
        if response.status_code == 200:
            df = pd.json_normalize(response.json())
            df.to_csv("spacex_launches.csv", index=False)
        else:
            st.error(f"Failed to fetch data from SpaceX API. Status code: {response.status_code}")
            return None

    # Drop unnecessary columns
    unnecessary_columns = [
        'launch_library_id', 'id', 'fairings.ships', 'links.patch.small', 'links.patch.large',
        'links.reddit.campaign', 'links.reddit.launch', 'links.reddit.media', 'links.reddit.recovery',
        'links.presskit', 'links.webcast', 'links.youtube_id', 'links.article', 'links.wikipedia',
        'static_fire_date_unix', 'auto_update', 'ships', 'capsules', 'links.flickr.small',
        'links.flickr.original', 'fairings'
    ]
    df = df.drop(columns=[col for col in unnecessary_columns if col in df.columns], errors='ignore')

    # Convert data types
    for col in ['success', 'fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered']:
        if col in df.columns:
            df[col] = df[col].map({'true': True, 'false': False, True: True, False: False, None: None}, na_action='ignore').astype('boolean')
    for col in ['date_utc', 'static_fire_date_utc']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Handle missing values
    if 'success' in df.columns and 'upcoming' in df.columns:
        df.loc[df['upcoming'] == True, 'success'] = df.loc[df['upcoming'] == True, 'success'].fillna(False)
    if 'success' in df.columns:
        df['success'] = df['success'].fillna(False)
    for col in ['fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered']:
        if col in df.columns:
            df[col] = df[col].fillna(False)
    if 'window' in df.columns:
        df['window'] = df['window'].fillna(df['window'].median())
    if 'core_flight' in df.columns:
        df['core_flight'] = df['core_flight'].fillna(0)
    categorical_columns = ['details', 'date_precision', 'name', 'crew', 'failures']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Simplify complex columns
    if 'cores' in df.columns and 'core_flight' not in df.columns:
        df['core_flight'] = df['cores'].apply(lambda x: ast.literal_eval(x)[0]['flight'] if isinstance(x, str) and x and ast.literal_eval(x) else 0).astype('float64')
        df = df.drop(columns=['cores'], errors='ignore')
    elif 'core_flight' not in df.columns:
        df['core_flight'] = 0.0
    if 'payloads' in df.columns and 'payload_id' not in df.columns:
        df['payload_id'] = df['payloads'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x and ast.literal_eval(x) else 'Unknown')
        df = df.drop(columns=['payloads'], errors='ignore')
    elif 'payload_id' not in df.columns:
        df['payload_id'] = 'Unknown'

    # Map rocket and launchpad names
    if 'rocket' in df.columns and 'rocket_name' not in df.columns:
        rocket_dict = get_lookup_dict("https://api.spacexdata.com/v4/rockets")
        df['rocket_name'] = df['rocket'].map(rocket_dict).fillna('Unknown')
    if 'launchpad' in df.columns and 'launchpad_name' not in df.columns:
        launchpad_dict = get_lookup_dict("https://api.spacexdata.com/v4/launchpads")
        df['launchpad_name'] = df['launchpad'].map(launchpad_dict).fillna('Unknown')

    # Drop redundant ID columns after mapping
    df = df.drop(columns=['rocket', 'launchpad'], errors='ignore')

    # Add payload mass
    payloads = requests.get("https://api.spacexdata.com/v4/payloads").json()
    payload_mass_map = {payload["id"]: payload.get("mass_kg", None) for payload in payloads}
    df["mass_kg"] = df["payload_id"].map(payload_mass_map)
    df['mass_kg'] = df['mass_kg'].fillna(df['mass_kg'].mean()).astype(float)

    # Add launchpad coordinates
    launchpad_df = pd.DataFrame(requests.get("https://api.spacexdata.com/v4/launchpads").json())
    launchpad_df = launchpad_df[['id', 'name', 'latitude', 'longitude']].rename(columns={'id': 'launchpad', 'name': 'launchpad_name'})
    df = df.merge(launchpad_df[['launchpad_name', 'latitude', 'longitude']], on='launchpad_name', how='left')

    # Add year for filtering
    df['year'] = pd.to_datetime(df['date_utc']).dt.year

    return df

# Load data
df = load_and_preprocess_data()
if df is None:
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")
year_options = ['All'] + sorted(df['year'].unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", year_options)
rocket_options = ['All'] + sorted(df['rocket_name'].unique().tolist())
selected_rocket = st.sidebar.selectbox("Select Rocket Type", rocket_options)
launchpad_options = ['All'] + sorted(df['launchpad_name'].unique().tolist())
selected_launchpad = st.sidebar.selectbox("Select Launch Site", launchpad_options)

# Filter DataFrame
filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
if selected_rocket != 'All':
    filtered_df = filtered_df[filtered_df['rocket_name'] == selected_rocket]
if selected_launchpad != 'All':
    filtered_df = filtered_df[filtered_df['launchpad_name'] == selected_launchpad]

# Data Overview
st.header("📊 Historical Launch Data")
st.write(f"Showing {len(filtered_df)} launches")
if len(filtered_df) > 0:
    st.dataframe(filtered_df[[
        'flight_number', 'name', 'date_utc', 'rocket_name', 'launchpad_name', 'success', 'mass_kg', 'details'
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
        if filtered_df['mass_kg'].notna().any() and filtered_df['success'].nunique() > 1:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_df, x='success', y='mass_kg', ax=ax)
            ax.set_title('Payload Mass vs. Launch Success')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data or single success category for Payload Mass vs. Success plot.")
    else:
        st.warning("No data available for Payload Mass vs. Success plot.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Rocket Type vs. Success")
    if len(filtered_df) > 0 and 'rocket_name' in filtered_df.columns and 'success' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='rocket_name', hue='success', ax=ax)
        ax.set_title('Rocket Type vs. Launch Success')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for Rocket Type vs. Success plot.")

with col4:
    st.subheader("Launches per Year")
    if len(filtered_df) > 0 and 'year' in filtered_df.columns and 'success' in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x='year', hue='success', ax=ax)
        ax.set_title('Launches per Year by Success')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for Launches per Year plot.")

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
    ml_df = df[['success', 'rocket_name', 'launchpad_name', 'mass_kg', 'window', 'core_flight', 'fairings.reused', 'fairings.recovery_attempt', 'fairings.recovered']].copy()
    ml_df['success'] = ml_df['success'].astype(int)
    
    num_cols = ['mass_kg', 'window', 'core_flight']
    imputer = SimpleImputer(strategy='median')
    ml_df[num_cols] = imputer.fit_transform(ml_df[num_cols])
    
    cat_cols = ['rocket_name', 'launchpad_name']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
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
    
    return rf_model, label_encoders, imputer

model, label_encoders, imputer = train_ml_model()

# Prediction input form
st.subheader("Input Launch Parameters")
with st.form("prediction_form"):
    rocket_name = st.selectbox("Rocket Type", sorted(df['rocket_name'].unique()))
    launchpad_name = st.selectbox("Launch Site", sorted(df['launchpad_name'].unique()))
    mass_kg = st.slider("Payload Mass (kg)", min_value=0.0, max_value=float(df['mass_kg'].max()), value=float(df['mass_kg'].mean()))
    window = st.slider("Launch Window (seconds)", min_value=0.0, max_value=float(df['window'].max()), value=float(df['window'].median()))
    core_flight = st.number_input("Core Flight Number", min_value=0, value=1)
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
        'fairings.recovered': [fairings_recovered]
    })
    
    for col in ['rocket_name', 'launchpad_name']:
        input_data[col] = label_encoders[col].transform(input_data[col])
    input_data[['mass_kg', 'window', 'core_flight']] = imputer.transform(input_data[['mass_kg', 'window', 'core_flight']])
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
**Data Source**: SpaceX API  
**Deployment**: Run locally with `streamlit run app.py` or deploy on Heroku/Render.  
**Note**: Weather data integration is planned for future updates.
""")