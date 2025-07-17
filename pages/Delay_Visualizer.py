import os
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from objects.Delay_network import (
    DelayBubbleMap, DelayBubbleMap2, DelayHeatmap,
    DelayHourlyTotalLineChart, DelayHourlyTotalLineChartByTrain, DelayHourlyLinkTotalLineChart
)
from objects.Boxplot import DelayBoxPlot, StationBoxPlot

# Paths
MART_PATH = os.getenv("MART_RELATIVE_PATH")
STATIONS_PATH = f"{MART_PATH}/public/stations.csv"
DELAY_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"
BOXPLOT_PATH = f"{MART_PATH}/public/df_monthly_with_headers.csv"
TRAIN_DATA_PATH = f"{MART_PATH}/public/cleaned_daily_full.csv"

st.set_page_config(page_title="🕒 Delay Visulizer", layout="wide")
st.title("🕒 Delay Visulizer")

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

@st.cache_resource
def get_station_chart():
    return DelayHourlyTotalLineChart(delay_data_path=DELAY_PATH)

@st.cache_resource
def get_train_chart():
    return DelayHourlyTotalLineChartByTrain(delay_data_path=TRAIN_DATA_PATH)

@st.cache_resource
def get_relation_chart():
    return DelayHourlyLinkTotalLineChart(delay_data_path=TRAIN_DATA_PATH)

stations_df = load_stations()
delays_df = load_delays()

clusters = {
    "Cluster 1 - Brussels": ["Bruxelles-Central", "Bruxelles-Midi", "Bruxelles-Nord"],
    "Cluster 2 - Antwerp": [
        "Anvers-Sud", "Anvers-Est", "Anvers-Luchtbal",
        "Anvers-Noorderdokken", "Anvers-Berchem",
        "Anvers-Central", "Anvers-Dam"]
}

if 'bubble_map' not in st.session_state:
    st.session_state.bubble_map = DelayBubbleMap(STATIONS_PATH, DELAY_PATH)
    st.session_state.bubble_map1 = DelayBubbleMap2(STATIONS_PATH, DELAY_PATH)
    st.session_state.heatmap = DelayHeatmap(DELAY_PATH)
    st.session_state.direction_box = DelayBoxPlot(BOXPLOT_PATH)
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)



if "station_box" not in st.session_state:
    st.session_state.station_box = StationBoxPlot(BOXPLOT_PATH)  # Choose a sensible default

# Now safe to access
station_box = st.session_state.station_box

bubble_map = st.session_state.bubble_map
bubble_map1 = st.session_state.bubble_map1
heatmap = st.session_state.heatmap
direction_box = st.session_state.direction_box
station_box = st.session_state.station_box

page = option_menu(
    menu_title=None,
    options=["Dashboard Tab", "Analytics Tab", "Hourly Delay Tab"],
    icons=["map", "bar-chart", "clock"],
    orientation="horizontal"
)

if page == "Dashboard Tab":
    selected_date = st.date_input("🗕️ Choose Day")
    st.markdown("### 🗘️ Bubble Map Filters")
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

    with st.expander("🔥 Delay Heatmaps (Top 10 Stations)"):
        st.markdown("### 🌟 Heatmap Filters")
        all_stations_in_data = sorted(stations_df["Name_FR"].dropna().astype(str).str.strip().str.title().unique())
        heatmap_cluster = st.radio("📍 Quick Select Cluster for Heatmaps", options=list(clusters.keys()), key="heatmap_cluster_selector")
        heatmap_default = clusters[heatmap_cluster]

        heatmap_stations = st.multiselect("Choose stations for heatmaps", options=all_stations_in_data, default=heatmap_default, key="heatmap_station_select")

        if st.button("Render Heatmaps"):
            heatmap.load_and_prepare(arrival=False, date_filter=selected_date)
            heatmap.filter_and_prepare_heatmap(arrival=False, station_filter=heatmap_stations)
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"#### Departure Heatmap for {selected_date.strftime('%Y-%m-%d')}")
                st.plotly_chart(heatmap.render_heatmap(arrival=False))

            heatmap.load_and_prepare(arrival=True, date_filter=selected_date)
            heatmap.filter_and_prepare_heatmap(arrival=True, station_filter=heatmap_stations)
            with col4:
                st.markdown(f"#### Arrival Heatmap for {selected_date.strftime('%Y-%m-%d')}")
                st.plotly_chart(heatmap.render_heatmap(arrival=True))

elif page == "Analytics Tab":
    with st.expander("📦 Total Delay Boxplot by Relation"):
        all_directions = sorted(d for d in direction_box.df["Relation direction"].dropna().unique() if "EURST" not in d)
        selected_directions = st.multiselect("Select up to 3 Relation Directions:", options=all_directions, max_selections=3)

        if selected_directions:
            all_related_dirs = []
            for d in selected_directions:
                relation = direction_box.get_relation_from_direction(d)
                if relation:
                    all_related_dirs.extend(direction_box.get_directions_by_relation(relation))
            all_related_dirs = list(set(all_related_dirs))

            st.markdown("### 🌟 Total Delay Distribution for Selected Relations")
            fig = direction_box.render_boxplot(directions=all_related_dirs)
            st.plotly_chart(fig, use_container_width=True)

            for d in selected_directions:
                st.markdown(f"### 🏢 Delay Distribution by Station for **{d}**")
                fig_station = direction_box.render_station_distribution_for_direction(d)
                if fig_station:
                    st.plotly_chart(fig_station, use_container_width=True)
                else:
                    st.info(f"No station-level data for {d}.")

elif page == "Hourly Delay Tab":
    st.subheader("📍 Delay by Station")
    df_delay = load_station_data(DELAY_PATH)
    df_train = load_train_data(TRAIN_DATA_PATH)

    all_stations = sorted(df_delay["Stopping place (FR)"].dropna().unique())
    selected_stations = st.multiselect("🌏 Select station(s):", options=all_stations, default=["Bruxelles-Midi"])

    if selected_stations:
        df_filtered = df_delay[df_delay["Stopping place (FR)"].isin(selected_stations)].copy()
        df_filtered["Total Delay"] = df_filtered["Delay at departure"].fillna(0) + df_filtered["Delay at arrival"].fillna(0)
        df_filtered = df_filtered[df_filtered["Total Delay"] > 0]
        df_filtered["Hour"] = pd.to_datetime(
            df_filtered["Actual departure time"].combine_first(df_filtered["Actual arrival time"]), errors="coerce"
        ).dt.hour
        hourly_totals = df_filtered.groupby(["Stopping place (FR)", "Hour"])["Total Delay"].sum().div(60).reset_index()

        total_delay = round(hourly_totals["Total Delay"].sum(), 1)
        avg_delay = round(hourly_totals["Total Delay"].mean(), 1)
        max_delay = round(hourly_totals["Total Delay"].max(), 1)
        count = len(hourly_totals)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🕒 Total Delay (min)", f"{total_delay}")
        col2.metric("📈 Average Delay (min)", f"{avg_delay}")
        col3.metric("🚨 Max Delay (min)", f"{max_delay}")
        col4.metric("🗾️ Hourly Records", f"{count}")

        fig_station = get_station_chart().plot(selected_stations=selected_stations)
        if fig_station:
            st.plotly_chart(fig_station, use_container_width=True)
        else:
            st.warning("No delay data available for the selected stations.")

    st.subheader("🚆 Delay by Train Number")
    all_trains = sorted(df_train["Train number"].unique())
    selected_trains = st.multiselect("🚂 Select Train Number(s):", options=all_trains, default=[all_trains[0]] if all_trains else [])

    if selected_trains:
        fig_train, grouped_data = get_train_chart().plot(selected_trains=selected_trains, return_data=True)

        if not grouped_data.empty:
            total_delay = round(grouped_data["Total Delay (min)"].sum(), 1)
            avg_delay = round(grouped_data["Total Delay (min)"].mean(), 1)
            max_delay = round(grouped_data["Total Delay (min)"].max(), 1)
            count = len(grouped_data)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🕒 Total Delay (min)", f"{total_delay}")
            col2.metric("📊 Avg Delay (min)", f"{avg_delay}")
            col3.metric("🚨 Max Hourly Delay (min)", f"{max_delay}")
            col4.metric("🔢 Records", f"{count}")

            st.plotly_chart(fig_train, use_container_width=True)
        else:
            st.warning("No delay data for selected train numbers.")

    st.subheader("🔁 Delay by Relation Direction")
    relation_chart = get_relation_chart()
    all_relations = sorted(relation_chart.df["Relation direction"].dropna().unique().tolist())
    selected_relations = st.multiselect("🧽 Select Relation Direction(s):", options=all_relations, default=all_relations[:3])

    if selected_relations:
        fig_relation, grouped_relation = relation_chart.plot_by_relation_direction(selected_relations=selected_relations, return_data=True)

        total_delay_rel = round(grouped_relation["Total Delay"].sum(), 1)
        avg_delay_rel = round(grouped_relation["Total Delay"].mean(), 1)
        max_delay_rel = round(grouped_relation["Total Delay"].max(), 1)
        count_rel = len(grouped_relation)

        colr1, colr2, colr3, colr4 = st.columns(4)
        colr1.metric("🕒 Total Delay (min)", f"{total_delay_rel}")
        colr2.metric("📈 Average Delay (min)", f"{avg_delay_rel}")
        colr3.metric("🚨 Max Delay (min)", f"{max_delay_rel}")
        colr4.metric("🗾️ Hourly Records", f"{count_rel}")

        st.plotly_chart(fig_relation, use_container_width=True)
    else:
        st.warning("Please select at least one relation direction to display the chart.")
