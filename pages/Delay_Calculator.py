"""
Delay_Calculator.py

Streamlit dashboard for visualizing total delay by hour for selected stations.

Author: Mohamad Hussain
Date: [2025-06-26]
"""

import streamlit as st
import pandas as pd
import os
from objects.Delay_network import DelayHourlyTotalLineChart

# Setup page
st.set_page_config(layout="wide")
st.title("📊 Total Delay by Hour Dashboard")

# Load delay data
DATA_PATH = f"{os.getenv('MART_RELATIVE_PATH')}/public/delays_standardized_titlecase.csv"
df_delay = pd.read_csv(DATA_PATH)

# Get all unique stations
df_delay["Stopping place (FR)"] = df_delay["Stopping place (FR)"].astype(str).str.title()
all_stations = sorted(df_delay["Stopping place (FR)"].dropna().unique())

# Station selector
selected_stations = st.multiselect(
    "🎯 Select station(s) to visualize total delay by hour:",
    options=all_stations,
    default=["Bruxelles-Midi"]
)

# If user selected any stations
if selected_stations:
    # Filter data
    df_filtered = df_delay[df_delay["Stopping place (FR)"].isin(selected_stations)].copy()

    # Calculate total delay (arrival + departure)
    df_filtered["Total Delay"] = df_filtered["Delay at departure"].fillna(0) + df_filtered["Delay at arrival"].fillna(0)
    df_filtered = df_filtered[df_filtered["Total Delay"] > 0]

    # Create hour column from first valid timestamp
    df_filtered["Hour"] = df_filtered["Actual departure time"].combine_first(df_filtered["Actual arrival time"])
    df_filtered["Hour"] = pd.to_datetime(df_filtered["Hour"], errors="coerce").dt.hour

    # Group by station and hour
    hourly_totals = (
        df_filtered.groupby(["Stopping place (FR)", "Hour"])["Total Delay"]
        .sum()
        .div(60)  # convert to minutes
        .reset_index()
        .rename(columns={"Total Delay": "Total Delay (min)"})
    )

    # Stats from grouped data
    total_delay = round(hourly_totals["Total Delay (min)"].sum(), 1)
    avg_delay = round(hourly_totals["Total Delay (min)"].mean(), 1)
    max_delay = round(hourly_totals["Total Delay (min)"].max(), 1)
    count = len(hourly_totals)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🕒 Total Delay (min)", f"{total_delay}")
    col2.metric("📈 Average Delay (min)", f"{avg_delay}")
    col3.metric("🚨 Max Delay (min)", f"{max_delay}")
    col4.metric("🧾 Hourly Records", f"{count}")

    # Draw chart using class
    chart = DelayHourlyTotalLineChart(delay_data_path=DATA_PATH)
    fig = chart.plot(selected_stations=selected_stations)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No delay data available for the selected stations.")
else:
    st.info("Please select at least one station to view delay data.")
