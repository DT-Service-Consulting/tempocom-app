import os
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from objects.Delay_network import (
    DelayBubbleMap, DelayBubbleMap2, DelayHeatmapDB,
    DelayHourlyTotalLineChart, DelayHourlyTotalLineChartByTrain, DelayHourlyLinkTotalLineChart
)
from objects.Boxplot import DelayBoxPlot,BaseBoxPlotDB, StationBoxPlot ,LinkBoxPlot
import time
# GLOBAL LIBRAIRIES
import importlib, sys, os
from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import count_distinct, col, length, avg, monotonically_increasing_id, row_number, concat_ws, try_to_timestamp
from pyspark.sql.functions import max as spark_max, min as spark_min, date_add, date_sub, to_date, date_format, upper, to_timestamp, when, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.window import Window
import pyodbc
import datetime
from requests import get, Response
from itertools import product
from pyspark.sql.utils import AnalysisException
from sqlalchemy.engine import Engine
from components import page_template


print(pyodbc.drivers())

# SETTING THE ENVIRONMENT
sys.path.append('../TEMPOCOM-APP')
load_dotenv('../tempocom_config/.env')

# LOCAL LIBRAIRIES
from utils import *
from modules import DBConnector

# INITIALIZATION
spark = (
    SparkSession.builder
        .appName("WriteToAzureSQL")
        # latest GA driver as of mid-2025 – change version if a newer one appears
        .config("spark.jars.packages", "com.microsoft.sqlserver:mssql-jdbc:10.2.1.jre11")
        .getOrCreate()
)


dbc = DBConnector()

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
    df = pd.read_csv(BOXPLOT_PATH)
    df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str).str.title()
    df["Relation direction"] = df["Relation direction"].astype(str)
    df["Train number"] = df["Train number"].astype(str)
    return df


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

if "bubble_map" not in st.session_state:
   st.session_state.bubble_map = DelayBubbleMap(dbc)

if "bubble_map1" not in st.session_state:
    st.session_state.bubble_map1 = DelayBubbleMap2(dbc)


# --- Ensure DB-backed boxplot objects exist in session state (create once) ---
if "direction_box" not in st.session_state:
    with st.spinner("Loading Relation (Direction) Boxplots from DB…"):
        st.session_state["direction_box"] = DelayBoxPlot(dbc)

if "station_box" not in st.session_state:
    with st.spinner("Loading Station Boxplots from DB…"):
        st.session_state["station_box"] = StationBoxPlot(dbc)

if "links_box" not in st.session_state:   # keep this key name to match the rest of your file
    with st.spinner("Loading Link Boxplots from DB…"):
        st.session_state["links_box"] = LinkBoxPlot(dbc)

# --- Local references (single assignment) ---
direction_box = st.session_state["direction_box"]
station_box   = st.session_state["station_box"]
link_box      = st.session_state["links_box"]


bubble_map = st.session_state.bubble_map
bubble_map1 = st.session_state.bubble_map1


page = option_menu(
    menu_title=None,
    options=["Dashboard Tab", "Analytics Tab", "Hourly Delay Tab"],
    icons=["map", "bar-chart", "clock"],
    orientation="horizontal"
)


####################################################################################################################################################
@st.cache_data
def stations_and_links_from_db(_dbc, selected_relation_names):
    if not selected_relation_names:
        return {}, {}

    placeholders = ",".join(["?"] * len(selected_relation_names))
    sql = f"""
        WITH rel AS (
            SELECT rd.ID, rd.name
            FROM relation_directions rd
            WHERE rd.name IN ({placeholders})
        ),
        ordered_stops AS (
            SELECT
                r.name AS relation_name,
                ds.order_in_route,
                op.Complete_name_in_French AS station_name
            FROM direction_stops ds
            JOIN rel r
              ON r.ID = ds.direction_id
            JOIN operational_points op
              ON ds.station_id = TRY_CAST(op.PTCAR_ID AS INT)
        )
        SELECT relation_name, station_name, order_in_route
        FROM ordered_stops
        ORDER BY relation_name, order_in_route;
    """
    st_df = pd.read_sql(sql, _dbc.conn, params=selected_relation_names)

    rel_to_stations = {}
    for rel, grp in st_df.groupby("relation_name", sort=False):
        seen, ordered = set(), []
        for nm in grp["station_name"]:
            nm = (nm or "").strip()
            if nm and nm not in seen:
                ordered.append(nm)
                seen.add(nm)
        rel_to_stations[rel] = ordered

    LINK_SEP = " ? "  # <-- match punctuality_boxplots_link.name
    rel_to_links = {}
    for rel, seq in rel_to_stations.items():
        links = [f"{a}{LINK_SEP}{b}" for a, b in zip(seq, seq[1:])]
        seen_l, ordered_l = set(), []
        for l in links:
            if l not in seen_l:
                ordered_l.append(l)
                seen_l.add(l)
        rel_to_links[rel] = ordered_l

    return rel_to_stations, rel_to_links


@st.cache_data
def load_all_stations(_conn):
    query = """
    SELECT DISTINCT Complete_name_in_French
    FROM operational_points
    WHERE Complete_name_in_French IS NOT NULL
    """
    raw_results = _conn.query(query)

    # Handle list of dicts like: [{'Short_name_in_French': 'Namur'}, ...]
    names = []
    for row in raw_results:
        try:
            raw_name = row.get('Complete_name_in_French', '')
            cleaned = " ".join(raw_name.strip().title().split())
            if cleaned:
                names.append(cleaned)
        except Exception as e:
            print(f"⚠️ Failed to clean row: {row} — {e}")

    return sorted(set(names))  # remove duplicates, sort alphabetically




import datetime
import streamlit as st
from objects.Delay_network import DelayBubbleMap

# --- Session state init ---
if 'bubble_map' not in st.session_state:
    st.session_state['bubble_map'] = DelayBubbleMap(dbc)

if 'bubble_map2' not in st.session_state:
    st.session_state['bubble_map2'] = DelayBubbleMap2(dbc)

if 'maps_ready' not in st.session_state:
    st.session_state['maps_ready'] = False

if 'station_list' not in st.session_state:
    st.session_state['station_list'] = load_all_stations(dbc)

if page == "Dashboard Tab":
    default_date = datetime.date(2024, 1, 16)
    selected_date = st.date_input("🗕️ Choose Day", value=default_date)

    # ----------------------- Bubble Map Section -----------------------
    st.markdown("## 🗘️ Bubble Map Filters")

    selected_cluster = st.radio("📍 Quick Select Cluster", options=list(clusters.keys()), key="bubble_cluster_selector")
    cluster_stations = clusters[selected_cluster]
    station_list = st.session_state['station_list']
    default_stations = [s for s in cluster_stations if s in station_list]

    selected_stations = st.multiselect(
        "🏢 Choose Stations for Bubble Maps",
        options=station_list,
        default=default_stations,
        key="bubble_map_station_select"
    )

    if st.button("🔁 Update Maps"):
        st.session_state['bubble_map'].prepare_data( arrival=True,
            station_filter=selected_stations,
            date_filter=selected_date
        )
        st.session_state['bubble_map2'].prepare_data(
    
            station_filter=selected_stations,
            date_filter=selected_date
        )
        st.session_state['maps_ready'] = True

    if st.session_state.get('maps_ready', False):
        st.markdown("### 🗺️ Delay Bubble Maps")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Arrival Delay")
            st_folium(
                st.session_state['bubble_map'].render_map(),
                width=600, height=500, key="arr_map"
            )

        with col2:
            st.markdown("#### Departure Delay")
            st_folium(
                st.session_state['bubble_map2'].render_map(),
                width=600, height=500, key="dep_map"
            )
    else:
        st.info("👆 Click '🔁 Update Maps' to load and display arrival/departure bubble maps.")


# ----------------------- Delay Heatmap Section -----------------------
    with st.expander("🔥 Delay Heatmaps (Top 10 Stations)"):
        st.markdown("### 🌟 Heatmap Filters")

        # Assumes stations_df is preloaded from operational_points
        all_stations_in_data = sorted(
            stations_df["Name_FR"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.title()
            .unique()
        )

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
            st.info("⏳ Querying database and rendering heatmaps...")

            heatmap_db = DelayHeatmapDB(dbc.conn, selected_date)

            col3, col4 = st.columns(2)

            # ---- DEPARTURE HEATMAP ----
            with col3:
                df_dep = heatmap_db.query_delay_data(station_filter=heatmap_stations)
                if not df_dep.empty:
                    pivot_dep = heatmap_db.create_pivot(df_dep)
                    fig_dep = heatmap_db.render_heatmap(pivot_dep, arrival=False)
                    st.markdown(f"#### Departure Heatmap for {selected_date.strftime('%Y-%m-%d')}")
                    st.plotly_chart(fig_dep)
                else:
                    st.warning("No departure delay data found for selected date/stations.")

            # ---- ARRIVAL HEATMAP ----
            with col4:
                df_arr = heatmap_db.query_delay_data(arrival=True, station_filter=heatmap_stations)
                if not df_arr.empty:
                    pivot_arr = heatmap_db.create_pivot(df_arr)
                    fig_arr = heatmap_db.render_heatmap(pivot_arr, arrival=True)
                    st.markdown(f"#### Arrival Heatmap for {selected_date.strftime('%Y-%m-%d')}")
                    st.plotly_chart(fig_arr)
                else:
                    st.warning("No arrival delay data found for selected date/stations.")




elif page == "Analytics Tab":
    with st.expander("📦 Total Delay Boxplots (Relation → Stations → Links)", expanded=True):
        all_relations = sorted(direction_box.df["name"].dropna().unique())
        selected_relations = st.multiselect(
            "Select relation direction(s):",
            options=all_relations,
            max_selections=3
        )

        if not selected_relations:
            st.info("Select at least one relation to see the boxplots.")
            st.stop()

        st.markdown("### 🎯 Total Delay — Selected Relation(s)")
        st.plotly_chart(direction_box.render_boxplot(filter_names=selected_relations), use_container_width=True)

        rel_to_stations, rel_to_links = stations_and_links_from_db(dbc, selected_relations)
        stations_for_rel = sorted({s for rel in selected_relations for s in rel_to_stations.get(rel, [])})
        links_for_rel    = sorted({l for rel in selected_relations for l in rel_to_links.get(rel, [])})

        st.markdown("### 🏢 Total Delay — Stations on Selected Relation(s)")
        if stations_for_rel:
            st.plotly_chart(station_box.render_boxplot(selected_directions=selected_relations), use_container_width=True)
        else:
            st.info("No stations found for the selected relation(s) via direction_stops.")

        st.markdown("### 🔗 Total Delay — Links (Consecutive Stations) on Selected Relation(s)")
        if links_for_rel:
            st.plotly_chart(link_box.render_boxplot(filter_names=links_for_rel), use_container_width=True)
        else:
            st.info("No links found for the selected relation(s) via direction_stops.")




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
