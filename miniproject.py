import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Traffic Accident Hotspot Analysis", layout="wide")

@st.cache_data
def load_data():

    data = {
        'latitude': np.random.uniform(18.5, 28.7, 100),

        'longitude': np.random.uniform(72.8, 77.2, 100),
        'severity': np.random.choice(['Minor', 'Major', 'Fatal'], 100),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 100),
        'weather': np.random.choice(['Clear', 'Rainy', 'Foggy'], 100)
    }
    return pd.DataFrame(data)

st.title("🚦 Traffic Accident Hotspot Analysis")
st.markdown("""**Project by:** Kailash Sharma  
**Objective:** Identify high-risk geographical locations and visualize accident trends.""")
st.write("---")

st.sidebar.header("Filter Options")
uploaded_file = st.sidebar.file_uploader("Upload Accident Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("Dataset must contain 'latitude' and 'longitude' columns.")
        st.stop()
else:
    st.sidebar.info("Using generated dummy data for demonstration.")
    df = load_data()

with st.expander("View Raw Data"):
    st.dataframe(df.head())

st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig_sev = px.bar(df, x='severity', title="Accident Counts by Severity", color='severity')
    st.plotly_chart(fig_sev, use_container_width=True)

with col2:
    if 'time_of_day' in df.columns:
        fig_time = px.pie(df, names='time_of_day', title="Accidents by Time of Day")
        st.plotly_chart(fig_time, use_container_width=True)

st.subheader("📍 Hotspot Identification (K-Means Clustering)")

num_clusters = st.sidebar.slider("Number of Hotspots (Clusters)", 2, 10, 5)

X = df[['latitude', 'longitude']].dropna()

if len(X) > num_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    st.success(f"Identified {num_clusters} accident hotspots using K-Means clustering.")
else:
    st.warning("Not enough data points for clustering.")


st.subheader("🗺️ Geospatial Hotspot Map")

m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)

for idx, row in df.iterrows():
    color = 'red' if row.get('severity') == 'Fatal' else 'blue'

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"Severity: {row.get('severity', 'N/A')}"
    ).add_to(m)

if 'cluster' in df.columns:
    for center in centers:
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color='black', icon='info-sign', prefix='fa'),
            tooltip="High Risk Hotspot Center"
        ).add_to(m)

st_folium(m, width=800, height=500)

st.subheader("📢 Actionable Insights")
st.markdown("""
* **High Risk Zones:** The black markers on the map indicate mathematically calculated centers of accident clusters.
* **Targeted Action:** Traffic authorities should prioritize these zones for patrol deployment[cite: 102].
* **Infrastructure:** Engineers should inspect these specific lat/long coordinates for road defects[cite: 104].
""")