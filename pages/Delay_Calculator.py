import streamlit as st
import pandas as pd
import os
from objects.Delay_network import DelayHourlyTotalLineChart, DelayHourlyTotalLineChartByTrain,DelayHourlyLinkTotalLineChart

# Setup page
st.set_page_config(layout="wide")
st.title("📊 Total Delay by Hour Dashboard")

# Paths to data
DATA_PATH = f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv"
TRAIN_DATA_PATH = f"{os.getenv('MART_RELATIVE_PATH')}/public/cleaned_daily_full.csv"

# Load station delay data
df_delay = pd.read_csv(DATA_PATH)
df_delay["Stopping place (FR)"] = (
    df_delay["Stopping place (FR)"]
    .astype(str)
    .str.title()
)


# --- STATION DELAY CHART ---
st.header("📍 Delay by Station")
all_stations = sorted(df_delay["Stopping place (FR)"].dropna().unique())
selected_stations = st.multiselect(
    "🎯 Select station(s):",
    options=all_stations,
    default=["Bruxelles-Midi"]
)

if selected_stations:
    df_filtered = df_delay[df_delay["Stopping place (FR)"].isin(selected_stations)].copy()
    df_filtered["Total Delay"] = (
        df_filtered["Delay at departure"].fillna(0)
        + df_filtered["Delay at arrival"].fillna(0)
    )
    df_filtered = df_filtered[df_filtered["Total Delay"] > 0]
    df_filtered["Hour"] = (
        df_filtered["Actual departure time"]
        .combine_first(df_filtered["Actual arrival time"])
    )
    df_filtered["Hour"] = (
        pd.to_datetime(df_filtered["Hour"], errors="coerce")
        .dt.hour
    )

    hourly_totals = (
        df_filtered
        .groupby(["Stopping place (FR)", "Hour"])["Total Delay"]
        .sum()
        .div(60)
        .reset_index()
    )

    total_delay = round(hourly_totals["Total Delay"].sum(), 1)
    avg_delay = round(hourly_totals["Total Delay"].mean(), 1)
    max_delay = round(hourly_totals["Total Delay"].max(), 1)
    count = len(hourly_totals)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🕒 Total Delay (min)", f"{total_delay}")
    col2.metric("📈 Average Delay (min)", f"{avg_delay}")
    col3.metric("🚨 Max Delay (min)", f"{max_delay}")
    col4.metric("🧾 Hourly Records", f"{count}")

    chart_station = DelayHourlyTotalLineChart(delay_data_path=DATA_PATH)
    fig_station = chart_station.plot(selected_stations=selected_stations)
    if fig_station:
        st.plotly_chart(fig_station, use_container_width=True)
    else:
        st.warning("No delay data available for the selected stations.")

# --- TRAIN DELAY CHART ---
st.header("🚆 Delay by Train Number")

# Load train delay data
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_train["Train number"] = df_train["Train number"].astype(str)

all_trains = sorted(df_train["Train number"].unique())
selected_trains = st.multiselect(
    "🚂 Select Train Number(s):",
    options=all_trains,
    default=[all_trains[0]] if all_trains else []
)

if selected_trains:
    chart_train = DelayHourlyTotalLineChartByTrain(delay_data_path=TRAIN_DATA_PATH)
    fig_train, grouped_data = chart_train.plot(
        selected_trains=selected_trains,
        return_data=True
    )

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


# === Relation Direction Section ===

st.markdown("## 🔁 Total Delay by Hour (Relation Direction)")

RELATION_PATH = f"{os.getenv('MART_RELATIVE_PATH')}/public/cleaned_daily_full.csv"
relation_chart = DelayHourlyLinkTotalLineChart(delay_data_path=RELATION_PATH)

# Load all unique directions
all_relations = sorted(relation_chart.df["Relation direction"].dropna().unique().tolist())

# UI filter
selected_relations = st.multiselect(
    "🧭 Select Relation Direction(s):",
    options=all_relations,
    default=all_relations[:3]  # or [] if you want it empty by default
)

# Plot and Stats
if selected_relations:
    fig_relation, grouped_relation = relation_chart.plot_by_relation_direction(
        selected_relations=selected_relations,
        return_data=True
    )

    # Calculate metrics
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
