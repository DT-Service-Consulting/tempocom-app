"""
Domino Effect Analyzer Streamlit App

This Streamlit application provides interactive visualizations for analyzing train delays and their domino effects across the network.

Features
--------
- Bubble maps of total arrival and departure delays per station (Folium)
- Heatmaps of delays by station and hour (Plotly)
- Boxplots of total delay by train direction and by station (Plotly)

Workflow
--------
1. Loads and filters delay and station data from CSV files.
2. Displays interactive bubble maps for arrival and departure delays.
3. Shows heatmaps for the top 10 stations by total delay.
4. Provides boxplots for exploring delay distributions by direction and by station, with user-selectable filters.

Dependencies
------------
- streamlit
- streamlit_folium
- pandas
- plotly
- folium
- streamlit-option-menu

Typical Usage
-------------
Run this script as a Streamlit app to explore and analyze train delay patterns and their network-wide effects.

Example:
    streamlit run Domino_Effect_Analyzer.py
"""

import os
import streamlit as st

import pandas as pd

from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2, DelayHeatmap
from objects.Boxplot import DelayBoxPlot, StationBoxPlot

# Paths
MART_PATH = os.getenv("MART_RELATIVE_PATH")
STATIONS_PATH = f"{MART_PATH}/public/stations.csv"
DELAY_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"
BOXPLOT_PATH = f"{MART_PATH}/public/df_monthly_with_headers.csv"

st.set_page_config(page_title="Domino Effect Analyzer", layout="wide")
st.title("🌊 Domino Effect Analyzer")

# Caching
@st.cache_data
def load_stations():
    return pd.read_csv(
        STATIONS_PATH,
        usecols=["Name_FR", "Geo_Point"]
    )

@st.cache_data
def load_delays():
    return pd.read_csv(
        DELAY_PATH,
        usecols=[
            "Stopping place (FR)", "Actual arrival time", "Actual departure time",
            "Delay at arrival", "Delay at departure"
        ],
        parse_dates=["Actual arrival time", "Actual departure time"],
        dtype={
            "Stopping place (FR)": "category",
            "Delay at arrival": "float32",
            "Delay at departure": "float32"
        }
    )


@st.cache_data
def load_boxplot_data(): return pd.read_csv(BOXPLOT_PATH)

stations_df = load_stations()
delays_df = load_delays()

# Cluster definitions
clusters = {
    "Cluster 1 - Brussels": ["Bruxelles-Central", "Bruxelles-Midi", "Bruxelles-Nord"],
    "Cluster 2 - Antwerp": [
        "Anvers-Sud", "Anvers-Est", "Anvers-Luchtbal",
        "Anvers-Noorderdokken", "Anvers-Berchem",
        "Anvers-Central", "Anvers-Dam"
    ]

}

# Session state
if 'bubble_map' not in st.session_state:
    st.session_state.bubble_map = DelayBubbleMap(STATIONS_PATH, DELAY_PATH)
    st.session_state.bubble_map1 = DelayBubbleMap2(STATIONS_PATH, DELAY_PATH)
    st.session_state.heatmap = DelayHeatmap(DELAY_PATH)
    st.session_state.direction_box = DelayBoxPlot(BOXPLOT_PATH)
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)

bubble_map = st.session_state.bubble_map
bubble_map1 = st.session_state.bubble_map1
heatmap = st.session_state.heatmap
direction_box = st.session_state.direction_box
station_box = st.session_state.station_box

# Navigation
page = option_menu(
    menu_title=None,
    options=["Maps & Heatmaps", "Boxplots"],
    icons=["map", "bar-chart"],
    orientation="horizontal"
)

# === MAPS & HEATMAPS ===
if page == "Maps & Heatmaps":
    selected_date = st.date_input("📅 Choose Day")

    # -------------------
    # 🗺️ Bubble Map Filters
    # -------------------
    st.markdown("### 🗺️ Bubble Map Filters")
    selected_cluster = st.radio("📍 Quick Select Cluster", options=list(clusters.keys()), key="bubble_cluster_selector")
    cluster_stations = clusters[selected_cluster]

    all_stations_in_data = sorted(pd.read_csv(STATIONS_PATH)["Name_FR"].dropna().unique())
    bubble_map_stations = st.multiselect(
        "🏢 Choose Stations for Bubble Maps",
        options=all_stations_in_data,
        default=cluster_stations,
        key="bubble_map_station_select"
    )

    if st.button("🔁 Update Maps"):
        bubble_map.prepare_data(station_filter=bubble_map_stations, date_filter=selected_date)
        bubble_map1.prepare_data1(station_filter=bubble_map_stations, date_filter=selected_date)
        st.session_state.maps_ready = True

    if st.session_state.get("maps_ready"):
        st.subheader("📍 Delay Bubble Maps")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Departure Delays")
            st_folium(bubble_map1.render_map1(), width=700, height=500, key="dep_map")
        with col2:
            st.markdown("#### Arrival Delays")
            st_folium(bubble_map.render_map(), width=700, height=500, key="arr_map")

    # -------------------
    # 🔥 Heatmap Filters
    # -------------------
    # -------------------
# 🔥 Heatmap Filters
# -------------------
with st.expander("🔥 Delay Heatmaps (Top 10 Stations)"):
    st.markdown("### 🎯 Heatmap Filters")

    heatmap_cluster = st.radio(
        "📍 Quick Select Cluster for Heatmaps",
        options=list(clusters.keys()),
        key="heatmap_cluster_selector"
    )
    heatmap_default = clusters[heatmap_cluster]

    heatmap_stations = st.multiselect(
        "Choose stations for heatmaps",
        options=all_stations_in_data,
        default=heatmap_default,
        key="heatmap_station_select"
    )

    if st.button("Render Heatmaps"):
        # 🟡 Apply selected date here
        heatmap.load_and_prepare(arrival=False, date_filter=selected_date)
        heatmap.filter_and_prepare_heatmap(arrival=False, station_filter=heatmap_stations)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"#### Departure Heatmap for {selected_date.strftime('%Y-%m-%d')}")
            st.plotly_chart(heatmap.render_heatmap(arrival=False))

        # 🔁 Repeat for arrival
        heatmap.load_and_prepare(arrival=True, date_filter=selected_date)
        heatmap.filter_and_prepare_heatmap(arrival=True, station_filter=heatmap_stations)

        with col4:
            st.markdown(f"#### Arrival Heatmap for {selected_date.strftime('%Y-%m-%d')}")
            st.plotly_chart(heatmap.render_heatmap(arrival=True))


# === BOXPLOTS ===
if page == "Boxplots":
    with st.expander("📦 Total Delay Boxplot by Relation"):
        all_directions = sorted(direction_box.df["Relation direction"].dropna().unique())
        selected_direction = st.selectbox("Select a Relation direction:", all_directions)

        if st.button("Show Boxplot for This Relation"):
            relation = direction_box.get_relation_from_direction(selected_direction)
            if relation:
                related_dirs = direction_box.get_directions_by_relation(relation)
                st.markdown(f"Showing all directions for Relation **{relation}**")
                st.plotly_chart(direction_box.render_boxplot(directions=related_dirs), use_container_width=True)
                fig_station = direction_box.render_station_distribution_for_direction(selected_direction)
                if fig_station:
                    st.markdown(f"### 🏢 Delay Distribution by Station for **{selected_direction}**")
                    st.plotly_chart(fig_station, use_container_width=True)
                else:
                    st.info("No station-level data available for this direction.")
            else:
                st.warning("⚠️ No Relation found for the selected direction.")
