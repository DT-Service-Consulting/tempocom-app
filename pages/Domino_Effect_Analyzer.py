import os
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2, DelayHeatmap

# -------------------------------
# 🌐 Constants
# -------------------------------
MART_PATH = os.getenv("MART_RELATIVE_PATH")
STATIONS_PATH = f"{MART_PATH}/public/stations.csv"
DELAY_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"

# -------------------------------
# 📍 Cluster Definitions
# -------------------------------
# Define station clusters
brussels_stations = [
    "Bruxelles-Central", "Bruxelles-Congrès", "Bruxelles-Chapelle",
    "Bruxelles-Midi", "Bruxelles-Nord", "Bruxelles-Schuman",
    "Bruxelles-Luxembourg", "Bruxelles-Ouest", "Schaerbeek", "Etterbeek",
    "Watermael", "Germoir", "Delta", "Meiser", "Bockstael", "Simonis",
    "Haren", "Haren-Sud", "Zaventem", "Nossegem", "Vilvorde", "Forest-Est",
    "Forest-Midi", "Uccle-Stalle", "Uccle-Calevoet", "Linkebeek", "Holleken",
    "Anderlecht", "Jette", "Berchem-Sainte-Agathe", "Boondael"
]

antwerp_stations = [
    "Anvers-Central", "Anvers-Berchem", "Anvers-Noorderdokken",
    "Anvers-Luchtbal", "Anvers-Est", "Anvers-Haven", "Anvers-Dam",
    "Ekeren", "Hoboken-Polder", "Mortsel", "Mortsel-Liersesteenweg",
    "Mortsel-Oude God", "Duffel", "Lierre", "Kontich-Lint", "Boechout",
    "Nijlen", "Noorderkempen", "Sint-Mariaburg", "Hemiksem", "Kalmthout",
    "Kapellen"
]
clusters = {
    "Brussels": brussels_stations,
    "Antwerp": antwerp_stations,
}

# -------------------------------
# 🚀 Streamlit Setup
# -------------------------------
st.set_page_config(page_title="Domino Effect Analyzer", layout="wide")
st.title("🌊 Domino Effect Analyzer")

# -------------------------------
# 📅 Input Section
# -------------------------------
selected_date = st.date_input("📅 Choose Day")

selected_cluster = st.radio("📍 Quick Station Cluster", options=["Brussels", "Antwerp", "Manual"])
stations_from_cluster = clusters.get(selected_cluster, [])
selected_stations = st.multiselect(
    "🚉 Choose Stations",
    options=sorted(pd.read_csv(STATIONS_PATH)["Name_FR"].unique()),
    default=stations_from_cluster
)

# -------------------------------
# 🔄 Prepare Visualization Objects
# -------------------------------
delay_df = pd.read_csv(DELAY_PATH)
stations_df = pd.read_csv(STATIONS_PATH)

# Convert date
delay_df["Date"] = pd.to_datetime(delay_df["Actual departure time"], errors="coerce").dt.date
filtered_delay_df = delay_df[delay_df["Date"] == selected_date]

# Save temporary filtered file
filtered_path = "/tmp/filtered_delays.csv"
filtered_delay_df.to_csv(filtered_path, index=False)

bubble_map = DelayBubbleMap(STATIONS_PATH, filtered_path)
bubble_map.prepare_data(station_filter=selected_stations)

bubble_map1 = DelayBubbleMap2(STATIONS_PATH, filtered_path)
bubble_map1.prepare_data1(station_filter=selected_stations)

heatmap = DelayHeatmap(filtered_path)
heatmap.load_and_prepare()
heatmap.load_and_prepare1()
heatmap.filter_and_prepare_heatmap(station_filter=selected_stations)
heatmap.filter_and_prepare_heatmap1(station_filter=selected_stations)

# -------------------------------
# 🗺️ Visualizations
# -------------------------------
st.subheader("📍 Delay Bubble Maps")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🚉 Departure Delays")
    st_folium(bubble_map1.render_map1(), width=700, height=500)

with col2:
    st.markdown("#### 🚉 Arrival Delays")
    st_folium(bubble_map.render_map(), width=700, height=500)

st.markdown("### 🔥 Delay Heatmaps")
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Departure")
    st.plotly_chart(heatmap.render_heatmap())

with col4:
    st.markdown("#### Arrival")
    st.plotly_chart(heatmap.render_heatmap1())
