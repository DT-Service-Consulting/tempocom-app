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
from components import page_template

# -------------------------------
# 🌟 Constants and Paths
# -------------------------------
MART_PATH = os.getenv("MART_RELATIVE_PATH")
STATIONS_PATH = f"{MART_PATH}/public/stations.csv"
DELAY_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"
BOXPLOT_PATH = f"{MART_PATH}/public/df_monthly_with_headers.csv"

# -------------------------------
# 📦 Page Setup
# -------------------------------
st.set_page_config(page_title="Domino Effect Analyzer", layout="wide")
st.title("🌊 Domino Effect Analyzer")

# -------------------------------
# 🚀 Caching Data
# -------------------------------
@st.cache_data
def load_stations():
    return pd.read_csv(STATIONS_PATH)

@st.cache_data
def load_delays():
    return pd.read_csv(DELAY_PATH)

@st.cache_data
def load_boxplot_data():
    return pd.read_csv(BOXPLOT_PATH)

# -------------------------------
# 🧠 Session State Object Init
# -------------------------------
if 'bubble_map' not in st.session_state:
    st.session_state.bubble_map = DelayBubbleMap(STATIONS_PATH, DELAY_PATH)
    st.session_state.bubble_map.prepare_data()

if 'bubble_map1' not in st.session_state:
    st.session_state.bubble_map1 = DelayBubbleMap2(STATIONS_PATH, DELAY_PATH)
    st.session_state.bubble_map1.prepare_data1()

if 'heatmap' not in st.session_state:
    st.session_state.heatmap = DelayHeatmap(DELAY_PATH)

if 'direction_box' not in st.session_state or 'station_box' not in st.session_state:
    st.session_state.direction_box = DelayBoxPlot(BOXPLOT_PATH)
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)

bubble_map = st.session_state.bubble_map
bubble_map1 = st.session_state.bubble_map1
heatmap = st.session_state.heatmap
direction_box = st.session_state.direction_box
station_box = st.session_state.station_box

# -------------------------------
# 🔀 Navigation Menu
# -------------------------------
page = option_menu(
    menu_title=None,
    options=["Maps & Heatmaps", "Boxplots"],
    icons=["map", "bar-chart"],
    orientation="horizontal",
    styles={
        "nav-link-selected": {"background-color": "#0e76a8", "font-weight": "bold"},
        "icon": {"color": "black", "font-size": "20px"},
    }
)

# -------------------------------
# 🗺️ MAPS & HEATMAPS PAGE
# -------------------------------
if page == "Maps & Heatmaps":
    stations_arrival = bubble_map.delay_summary['Stopping place (FR)'].tolist()
    stations_departure = bubble_map1.delay_summary['Stopping place (FR)'].tolist()
    all_stations = sorted(set(stations_arrival + stations_departure))

    selected_stations = st.multiselect(
        "Filter stations shown on the map:",
        options=all_stations,
        default=all_stations
    )

    if st.button("🔁 Update Maps"):
        bubble_map.prepare_data(station_filter=selected_stations)
        bubble_map1.prepare_data1(station_filter=selected_stations)
        st.session_state.maps_ready = True

    if st.session_state.get("maps_ready"):
        st.subheader("📍 Delay Bubble Maps")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Departure Delays")
            st_folium(bubble_map1.render_map1(), width=700, height=500)

        with col2:
            st.markdown("#### Arrival Delays")
            st_folium(bubble_map.render_map(), width=700, height=500)

    with st.expander("🔥 Delay Heatmaps (Top 10 Stations)"):
        if st.button("Render Heatmaps"):
            heatmap.load_and_prepare()
            heatmap.load_and_prepare1()
            heatmap.filter_and_prepare_heatmap()
            heatmap.filter_and_prepare_heatmap1()

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### Departure Delays")
                st.plotly_chart(heatmap.render_heatmap())

            with col4:
                st.markdown("#### Arrival Delays")
                st.plotly_chart(heatmap.render_heatmap1())

# -------------------------------
# 📦 BOXPLOTS PAGE
# -------------------------------
with st.expander("📦 Total Delay Boxplot by Relation"):
    st.markdown("Select a direction (e.g., 'BINCHE -> TURNHOUT') to show all directions of the same Relation (e.g., 'IC 11').")

    all_directions = sorted(direction_box.df["Relation direction"].dropna().unique())
    selected_direction = st.selectbox("Select a Relation direction:", all_directions, key="dir_select")

    if st.button("Show Boxplot for This Relation"):
        # Step 1: Get the Relation (e.g., IC 11)
        relation = direction_box.get_relation_from_direction(selected_direction)

        if relation:
            # Step 2: Get all directions under this Relation
            related_dirs = direction_box.get_directions_by_relation(relation)
            st.markdown(f"Showing all directions for Relation **{relation}**")

            st.plotly_chart(direction_box.render_boxplot(directions=related_dirs), use_container_width=True)
            # Additional station-level delay boxplot
            fig_station = direction_box.render_station_distribution_for_direction(selected_direction)
            if fig_station:
                st.markdown(f"### 🏢 Delay Distribution by Station for **{selected_direction}**")
                st.plotly_chart(fig_station, use_container_width=True)
            else:
                st.info("No station-level data available for this direction.")

        else:
            st.warning("⚠️ No Relation found for the selected direction.")
