"""
Delay_Calculator.py

Streamlit dashboard for visualizing train delay data using interactive maps and heatmaps.

Features:
- Loads and caches delay data for efficient visualization.
- Displays side-by-side bubble maps for departure and arrival delays at train stations.
- Allows users to filter stations shown on the maps via a multi-select widget.
- Shows heatmaps of delays for the top 10 stations (departure and arrival) using Plotly.
- Utilizes custom DelayBubbleMap, DelayBubbleMap2, and DelayHeatmap classes for data processing and visualization.

Environment Variables:
- Expects 'MART_RELATIVE_PATH' to be set for locating data files.

Dependencies:
- streamlit
- streamlit_folium
- plotly
- objects.Delay_network (custom module)

Author: Mohamad Hussain
Date: [2025-06-20]
"""



import streamlit as st
from streamlit_folium import st_folium
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2, DelayHeatmap
import os

# Set Streamlit page layout
st.set_page_config(layout="wide")
st.title("⏱️ Delay Visualization Dashboard")

# Load maps using cache
@st.cache_resource
def load_map():
    return DelayBubbleMap(
        stations_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/stations.csv",
        delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv"
    )

@st.cache_resource
def load_map1():
    return DelayBubbleMap2(
        stations_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/stations.csv",
        delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv"
    )

# Prepare data
bubble_map = load_map()
bubble_map.prepare_data()
stations_arrival = bubble_map.delay_summary['Stopping place (FR)'].tolist()

bubble_map1 = load_map1()
bubble_map1.prepare_data1()
stations_departure = bubble_map1.delay_summary['Stopping place (FR)'].tolist()

# Combine and deduplicate stations
all_top_stations = sorted(set(stations_arrival + stations_departure))

# Define station clusters
brussels_stations = [
    'Bruxelles-Central', 'Bruxelles-Congrès', 'Bruxelles-Chapelle',
    'Bruxelles-Midi', 'Bruxelles-Nord', 'Schaerbeek', 'Hal'
]
antwerp_stations = ['Anvers-Berchem', 'Anvers-Central', 'Malines']

# Advanced Station Filter UI
st.markdown("### 🎯 Station Filter")

# Station group selection (mutually exclusive)
station_group = st.radio(
    "Choose a station group to pre-select:",
    ["All", "Brussels", "Antwerp"],
    index=0,
    horizontal=True
)

# Determine initial selection based on checkboxes
if station_group == "All":
    initial_selection = all_top_stations
elif station_group == "Brussels":
    initial_selection = brussels_stations
elif station_group == "Antwerp":
    initial_selection = antwerp_stations
else:  # "None"
    initial_selection = []


# Filter to only valid stations
initial_selection = sorted(set(initial_selection).intersection(set(all_top_stations)))

# Multi-select dropdown
selected_stations = st.multiselect(
    "Manually add/remove stations:",
    options=all_top_stations,
    default=initial_selection
)

# Apply selected station filters
bubble_map.prepare_data(station_filter=selected_stations)
bubble_map1.prepare_data1(station_filter=selected_stations)

# Render Bubble Maps
st.markdown("### 📍 Delay Bubble Maps")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Departure Delays")
    m_departure = bubble_map1.render_map1()
    st_folium(m_departure, width=700, height=500)

with col2:
    st.markdown("#### Arrival Delays")
    m_arrival = bubble_map.render_map()
    st_folium(m_arrival, width=700, height=500)

# Delay Heatmaps
st.markdown("### 🔥 Delay Heatmaps (Top 10 Stations)")

heatmap = DelayHeatmap(
    delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv",
    top_n=10
)

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Departure Delays")
    fig_dep = heatmap.render_heatmap(delay_type="departure")
    st.plotly_chart(fig_dep)

with col4:
    st.markdown("#### Arrival Delays")
    fig_arr = heatmap.render_heatmap(delay_type="arrival")
    st.plotly_chart(fig_arr)
