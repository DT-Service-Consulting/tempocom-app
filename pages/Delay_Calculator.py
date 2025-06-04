import streamlit as st
from streamlit_folium import st_folium
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2

st.set_page_config(layout="centered")
st.title("⏱️ Delay Bubble Map")

@st.cache_resource
def load_map():
    return DelayBubbleMap(
        stations_path="./mart/public/stations.csv",
        delay_data_path="./mart/public/delays_standardized_titlecase.csv"
    )

@st.cache_resource
def load_map1():
    return DelayBubbleMap2(
        stations_path="./mart/public/stations.csv",
        delay_data_path="./mart/public/delays_standardized_titlecase.csv"
    )

# Load maps
bubble_map = load_map()
bubble_map.prepare_data()
stations_arrival = bubble_map.delay_summary['Stopping place (FR)'].tolist()

bubble_map1 = load_map1()
bubble_map1.prepare_data1()
stations_departure = bubble_map1.delay_summary['Stopping place (FR)'].tolist()

# Combine and deduplicate station names
all_top_stations = sorted(set(stations_arrival + stations_departure))

# UI: Multi-select filter
selected_stations = st.multiselect(
    "Filter stations shown on the map:",
    options=all_top_stations,
    default=all_top_stations
)

# Render filtered maps
st.subheader("Top 10 stations with delays at departure")
bubble_map1.prepare_data1(station_filter=selected_stations)
m2 = bubble_map1.render_map1()
st_folium(m2, width=700, height=400)

st.subheader("Top 10 stations with delays at arrival")
bubble_map.prepare_data(station_filter=selected_stations)
m1 = bubble_map.render_map()
st_folium(m1, width=700, height=400)

st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)

