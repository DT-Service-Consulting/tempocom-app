import streamlit as st
import pandas as pd
import os
from objects.Delay_network import (
    DelayHourlyTotalLineChart,
    DelayHourlyTotalLineChartByTrain,
    DelayHourlyLinkTotalLineChart
)

# Setup page
st.set_page_config(layout="wide")
st.title("📊 Total Delay by Hour Dashboard")

# File paths
MART_PATH = os.getenv("MART_RELATIVE_PATH")
DATA_PATH = f"{MART_PATH}/public/delays_standardized_titlecase.csv"
TRAIN_DATA_PATH = f"{MART_PATH}/public/cleaned_daily_full.csv"

# === Cached data loading ===
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

# === Cached class instances ===
@st.cache_resource
def get_station_chart():
    return DelayHourlyTotalLineChart(delay_data_path=DATA_PATH)

@st.cache_resource
def get_train_chart():
    return DelayHourlyTotalLineChartByTrain(delay_data_path=TRAIN_DATA_PATH)

@st.cache_resource
def get_relation_chart():
    return DelayHourlyLinkTotalLineChart(delay_data_path=TRAIN_DATA_PATH)

# === Load data ===
df_delay = load_station_data(DATA_PATH)
df_train = load_train_data(TRAIN_DATA_PATH)

# === STATION DELAY CHART ===
st.header("📍 Delay by Station")
all_stations = sorted(df_delay["Stopping place (FR)"].dropna().unique())
selected_stations = st.multiselect("🎯 Select station(s):", options=all_stations, default=["Bruxelles-Midi"])

if selected_stations:
    df_filtered = df_delay[df_delay["Stopping place (FR)"].isin(selected_stations)].copy()
    df_filtered["Total Delay"] = df_filtered["Delay at departure"].fillna(0) + df_filtered["Delay at arrival"].fillna(0)
    df_filtered = df_filtered[df_filtered["Total Delay"] > 0]

    df_filtered["Hour"] = pd.to_datetime(
        df_filtered["Actual departure time"].combine_first(df_filtered["Actual arrival time"]),
        errors="coerce"
    ).dt.hour

    hourly_totals = (
        df_filtered.groupby(["Stopping place (FR)", "Hour"])["Total Delay"]
        .sum().div(60).reset_index()
    )

    # Metrics
    total_delay = round(hourly_totals["Total Delay"].sum(), 1)
    avg_delay = round(hourly_totals["Total Delay"].mean(), 1)
    max_delay = round(hourly_totals["Total Delay"].max(), 1)
    count = len(hourly_totals)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🕒 Total Delay (min)", f"{total_delay}")
    col2.metric("📈 Average Delay (min)", f"{avg_delay}")
    col3.metric("🚨 Max Delay (min)", f"{max_delay}")
    col4.metric("🧾 Hourly Records", f"{count}")

    fig_station = get_station_chart().plot(selected_stations=selected_stations)
    if fig_station:
        st.plotly_chart(fig_station, use_container_width=True)
    else:
        st.warning("No delay data available for the selected stations.")

# === TRAIN DELAY CHART ===
st.header("🚆 Delay by Train Number")
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

# === RELATION DIRECTION CHART ===
st.header("🔁 Delay by Relation Direction")
relation_chart = get_relation_chart()
all_relations = sorted(relation_chart.df["Relation direction"].dropna().unique().tolist())

selected_relations = st.multiselect("🧭 Select Relation Direction(s):", options=all_relations, default=all_relations[:3])

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
    colr4.metric("🧾 Hourly Records", f"{count_rel}")

    st.plotly_chart(fig_relation, use_container_width=True)
else:
    st.warning("Please select at least one relation direction to display the chart.")
