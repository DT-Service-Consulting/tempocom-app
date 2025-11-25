import os
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from objects.Boxplot import DelayBoxPlot, StationBoxPlot ,LinkBoxPlot
from components.page_template import page_template

# Paths
MART_PATH = os.getenv("MART_RELATIVE_PATH")
STATIONS_PATH = f"{MART_PATH}/public/stations.csv"
DELAY_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"
BOXPLOT_PATH = f"{MART_PATH}/public/df_monthly_with_headers.csv"
TRAIN_DATA_PATH = f"{MART_PATH}/public/cleaned_daily_full.csv"

page_template("🕰️Delay Visualizer")

# Tutorial
st.info("""
**📚 How to use Delay Visualizer:**
1. **Bubble Map** : Visualize delays by station with colored bubbles
2. **Heatmap** : Analyze temporal distribution of delays
3. **Charts** : Explore hourly trends and by train
4. **Boxplots** : Compare delay distributions between stations and links
5. **Filters** : Use options to refine your analysis
""")


# Caching
@st.cache_data
def load_stations():
    return pd.read_csv(STATIONS_PATH, usecols=["Name_FR", "Geo_Point"])

@st.cache_data
def load_delays():
    return pd.read_csv(
        DELAY_PATH,
        usecols=["Stopping place (FR)", "Actual arrival time", "Actual departure time",
                 "Delay at arrival", "Delay at departure"],
        parse_dates=["Actual arrival time", "Actual departure time"],
        dtype={"Stopping place (FR)": "category", "Delay at arrival": "float32", "Delay at departure": "float32"}
    )

@st.cache_data
def load_boxplot_data():
    return pd.read_csv(BOXPLOT_PATH)

@st.cache_data
def load_station_data(path):
    df = pd.read_csv(path)
    df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str).str.title()
    return df

@st.cache_data
def load_train_data(path):
    df = pd.read_csv(path)
    df["Train number"] = df["Train number"].astype(str)
    return df


stations_df = load_stations()
delays_df = load_delays()

clusters = {
    "Cluster 1 - Brussels": ["Bruxelles-Central", "Bruxelles-Midi", "Bruxelles-Nord"],
    "Cluster 2 - Antwerp": [
        "Anvers-Sud", "Anvers-Est", "Anvers-Luchtbal",
        "Anvers-Noorderdokken", "Anvers-Berchem",
        "Anvers-Central", "Anvers-Dam"]
}


if "direction_box" not in st.session_state:
    st.session_state.direction_box = DelayBoxPlot(BOXPLOT_PATH)

if "station_box" not in st.session_state:
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)

if "links_box" not in st.session_state:
    st.session_state.links_box = LinkBoxPlot(BOXPLOT_PATH)



if "station_box" not in st.session_state:
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)  # Choose a sensible default

# Now safe to access
station_box = st.session_state.station_box

direction_box = st.session_state.direction_box
station_box = st.session_state.station_box
link_box = st.session_state.links_box



# Get all unique relation directions, filtering out EURST
all_directions = sorted(direction_box.df["Relation direction"].dropna().unique())


filtered_directions = [d for d in all_directions if "EURST" not in d]

selected_directions = st.multiselect(
    "Select up to 3 Relation Directions:",
    options=filtered_directions,
    max_selections=3
)

if selected_directions:
    # -- Get all related directions across selected relations
    all_related_dirs = set()
    for d in selected_directions:
        relation = direction_box.get_relation_from_direction(d)
        if relation:
            related = direction_box.get_directions_by_relation(relation)
            all_related_dirs.update(related)

    st.markdown("### 🎯 Total Delay Distribution for Selected Relations")
    fig = direction_box.render_boxplot(directions=list(all_related_dirs))
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No delay distribution data found for the selected directions.")

    # -- Per-direction breakdowns
    for d in selected_directions:
        st.markdown(f"### 🏢 Delay Distribution by Station for **{d}**")
        fig_station = direction_box.render_station_distribution_for_direction(d)
        if fig_station:
            st.plotly_chart(fig_station, use_container_width=True)
        else:
            st.info(f"No station-level data for **{d}**.")

        st.markdown(f"### 🔗 Delay Between Consecutive Stations in **{d}**")
        fig_link = link_box.render_boxplot(d)
        if fig_link:
            st.plotly_chart(fig_link, use_container_width=True)
        else:
            st.info(f"No link-level data for **{d}**.")

