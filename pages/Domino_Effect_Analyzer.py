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
st.markdown("### 📦 Delay Boxplot by Direction")

@st.cache_resource
def load_boxplot():
    return DelayBoxPlot(delay_data_path=f"{os.getenv('MART_RELATIVE_PATH')}/public/df_monthly_with_headers.csv")

boxplot = load_boxplot()

st.markdown(
    "Explore the **distribution of delays** (arrival or departure) across different train directions. "
    "You can switch delay types and filter directions."
)

col5, col6 = st.columns([1, 2])

with col5:
    delay_type = st.radio(
        "Delay Type:",
        options=["arrival", "departure"],
        horizontal=True
    )

with col6:
    top_directions = boxplot.df["Relation direction"].value_counts().nlargest(15).index.tolist()
    selected_directions = st.multiselect(
        "Select Directions:",
        options=top_directions,
        default=top_directions[1:5]
    )

if selected_directions:
    fig_box = boxplot.render_boxplot(
        delay_type=delay_type,
        directions=selected_directions
    )
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("⚠️ Please select at least one direction to view the boxplot.")
