import os
import streamlit as st
from streamlit_folium import st_folium
from objects.Delay_network import (
    DelayBubbleMap,
    DelayBubbleMap2,
    DelayHeatmap
)
from objects.Boxplot import DelayBoxPlot
from components import *

# Title and header
title = "🌊Domino Effect Analyzer"
page_template(title)

# -------------------------------
# Load Bubble Maps
# -------------------------------
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

bubble_map = load_map()
bubble_map.prepare_data()
stations_arrival = bubble_map.delay_summary['Stopping place (FR)'].tolist()

bubble_map1 = load_map1()
bubble_map1.prepare_data1()
stations_departure = bubble_map1.delay_summary['Stopping place (FR)'].tolist()

all_top_stations = sorted(set(stations_arrival + stations_departure))

# -------------------------------
# UI: Station Filter
# -------------------------------
selected_stations = st.multiselect(
    "Filter stations shown on the map:",
    options=all_top_stations,
    default=all_top_stations
)

bubble_map.prepare_data(station_filter=selected_stations)
bubble_map1.prepare_data1(station_filter=selected_stations)

# -------------------------------
# 📍 Delay Bubble Maps
# -------------------------------
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

# -------------------------------
# 🔥 Delay Heatmaps
# -------------------------------
st.markdown("### 🔥 Delay Heatmaps (Top 10 Stations)")

heatmap = DelayHeatmap(delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Departure Delays")
    heatmap.load_and_prepare()
    heatmap.filter_and_prepare_heatmap()
    fig_dep = heatmap.render_heatmap()
    st.plotly_chart(fig_dep)

with col4:
    st.markdown("#### Arrival Delays")
    heatmap.load_and_prepare1()
    heatmap.filter_and_prepare_heatmap1()
    fig_arr = heatmap.render_heatmap1()
    st.plotly_chart(fig_arr)

# -------------------------------
# 📦 Delay Boxplot by Direction
# -------------------------------
import os
import streamlit as st
from objects.Boxplot import DelayBoxPlot, StationBoxPlot

# Load both boxplot data sources
@st.cache_resource
def load_direction_boxplot():
    return DelayBoxPlot(
        delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/df_monthly_with_headers.csv"
    )

@st.cache_resource
def load_station_boxplot():
    return StationBoxPlot(
        delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/df_monthly_with_headers.csv"
    )

boxplot = load_direction_boxplot()
station_boxplot = load_station_boxplot()

# -------------------------------
# 📦 Total Delay Boxplot by Direction
# -------------------------------
st.markdown("### 📦 Total Delay Boxplot by Direction")

st.markdown(
    "Explore the **distribution of total delays** (sum of arrival + departure delays) "
    "across different train directions. This helps identify which directions have the highest variability or median delays."
)

# UI Controls
top_directions = boxplot.df["Relation direction"].value_counts().nlargest(15).index.tolist()

selected_directions = st.multiselect(
    "Select Directions:",
    options=top_directions,
    default=top_directions[1:5]
)

# Plot
if selected_directions:
    fig_box = boxplot.render_boxplot(
        directions=selected_directions
    )
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("⚠️ Please select at least one direction to view the boxplot.")

# -------------------------------
# 🏢 Total Delay Boxplot by Stopping Place
# -------------------------------
st.markdown("### 🏢 Total Delay Boxplot by Stopping Place")

st.markdown(
    "Visualize the **distribution of total delays by station**. You can choose which stations to include or let the app show the top ones automatically."
)

# UI Controls
top_stations = station_boxplot.df["Stopping place (FR)"].value_counts().nlargest(15).index.tolist()

selected_stations = st.multiselect(
    "Select Stations:",
    options=top_stations,
    default=top_stations[1:5]
)

# Plot
if selected_stations:
    fig_station_box = station_boxplot.render_boxplot(
        stations=selected_stations
    )
    st.plotly_chart(fig_station_box, use_container_width=True)
else:
    st.info("⚠️ Please select at least one station to view the boxplot.")
