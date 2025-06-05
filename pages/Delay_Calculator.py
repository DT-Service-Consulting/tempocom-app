import streamlit as st
from streamlit_folium import st_folium
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2
from objects.Delay_network import DelayHeatmap

st.set_page_config(layout="wide")
st.title("⏱️ Delay Visualization Dashboard")

# Load maps using cache
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

# Prepare data
bubble_map = load_map()
bubble_map.prepare_data()
stations_arrival = bubble_map.delay_summary['Stopping place (FR)'].tolist()

bubble_map1 = load_map1()
bubble_map1.prepare_data1()
stations_departure = bubble_map1.delay_summary['Stopping place (FR)'].tolist()

all_top_stations = sorted(set(stations_arrival + stations_departure))

# UI: Multi-select
selected_stations = st.multiselect(
    "Filter stations shown on the map:",
    options=all_top_stations,
    default=all_top_stations
)

# Apply filters
bubble_map.prepare_data(station_filter=selected_stations)
bubble_map1.prepare_data1(station_filter=selected_stations)

# Render side-by-side bubble maps
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

# Delay heatmaps side-by-side
st.markdown("### 🔥 Delay Heatmaps (Top 10 Stations)")

# Load heatmap data
heatmap = DelayHeatmap(delay_data_path="./mart/public/delays_standardized_titlecase.csv")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Departure Delays")
    heatmap.load_and_prepare()
    heatmap.filter_and_prepare_heatmap()
    fig_dep = heatmap.render_heatmap()
    st.pyplot(fig_dep)

with col4:
    st.markdown("#### Arrival Delays")
    heatmap.load_and_prepare1()
    heatmap.filter_and_prepare_heatmap1()
    fig_arr = heatmap.render_heatmap1()
    st.pyplot(fig_arr)
