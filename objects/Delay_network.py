"""
Delay_network.py

This module provides classes for visualizing train delay data using interactive bubble maps and heatmaps.

Classes:
- DelayBubbleMap: Visualizes total arrival delays per station as a Folium bubble map.
- DelayBubbleMap2: Visualizes total departure delays per station as a Folium bubble map.
- DelayHeatmap: Generates Plotly heatmaps of delays (arrival or departure) by station and hour.

Features:
- Reads station and delay data from CSV files.
- Aggregates and filters delay data for visualization.
- Supports filtering by station.
- Produces interactive maps and heatmaps for use in dashboards.

Dependencies:
- pandas
- folium
- ast
- matplotlib
- seaborn
- plotly

Author: Mohamad Hussain
Date: [2025-06-20]
"""


import pandas as pd
import plotly.express as px
import warnings
import pandas as pd
import folium
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class DelayBubbleMap:
    """
    Visualizes total arrival delays per station as a bubble map using Folium.
    """
    def __init__(self, stations_path: str, delay_data_path: str):
        """
        Initialize DelayBubbleMap with station and delay data CSV paths.

        Args:
            stations_path (str): Path to stations CSV file.
            delay_data_path (str): Path to delay data CSV file.
        """
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)
        self.prepare_data()

    def prepare_data(self, station_filter=None):
        """
        Prepare and merge arrival delay and station data.

        Args:
            station_filter (list, optional): List of station names to filter. Defaults to None.
        """
        # Filter out negative delays
        self.delays = self.delays[self.delays['Delay at arrival'] > 0]

        # Sum delay in minutes per station
        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at arrival']
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={'Delay at arrival': 'Total_Delay_Minutes'})
        )
        self.delay_summary = self.delay_summary.nlargest(10, 'Total_Delay_Minutes')

        # Title case for merging
        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].str.title()

        # Apply optional station filter
        if station_filter:
            station_filter = [s.title() for s in station_filter]
            self.delay_summary = self.delay_summary[self.delay_summary['Stopping place (FR)'].isin(station_filter)]

        # Merge with station geolocation
        self.merged = self.delay_summary.merge(
            self.stations[['Name_FR', 'Geo_Point']],
            left_on='Stopping place (FR)',
            right_on='Name_FR',
            how='inner'
        )

        # Convert string coordinates to tuples
        self.merged['Geo_Point'] = self.merged['Geo_Point'].apply(ast.literal_eval)

    def render_map(self):
        """
        Render a Folium bubble map of arrival delays.

        Returns:
            folium.Map: Folium map object with delay bubbles.
        """
        if len(self.merged) == 0:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        # Center the map based on average lat/lon
        lats = [pt[0] for pt in self.merged['Geo_Point']]
        lons = [pt[1] for pt in self.merged['Geo_Point']]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(location=(center_lat, center_lon), zoom_start=7, tiles="cartodb positron")

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            radius = min(max(delay_min / 10, 3), 15)

            popup = f"{row['Name_FR']}<br> Total Arrival Delay: {round(delay_min, 1)} min"

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color="crimson",
                fill=True,
                fill_color="orange",
                fill_opacity=0.7,
                popup=popup,
                tooltip=popup
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m


class DelayBubbleMap2:
    """
    Visualizes total departure delays per station as a bubble map using Folium.
    """
    def __init__(self, stations_path: str, delay_data_path: str):
        """
        Initialize DelayBubbleMap2 with station and delay data CSV paths.

        Args:
            stations_path (str): Path to stations CSV file.
            delay_data_path (str): Path to delay data CSV file.
        """
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)
        self.prepare_data1()

    def prepare_data1(self, station_filter=None):
        """
        Prepare and merge departure delay and station data.

        Args:
            station_filter (list, optional): List of station names to filter. Defaults to None.
        """
        self.delays = self.delays[self.delays['Delay at departure'] > 0]

        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at departure']
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={'Delay at departure': 'Total_Delay_Minutes'})
        )

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].str.title()

        self.delay_summary = self.delay_summary.nlargest(10, 'Total_Delay_Minutes')

        if station_filter:
            station_filter = [s.title() for s in station_filter]
            self.delay_summary = self.delay_summary[self.delay_summary['Stopping place (FR)'].isin(station_filter)]

        self.merged = self.delay_summary.merge(
            self.stations[['Name_FR', 'Geo_Point']],
            left_on='Stopping place (FR)',
            right_on='Name_FR',
            how='inner'
        )

        self.merged['Geo_Point'] = self.merged['Geo_Point'].apply(ast.literal_eval)

    def render_map1(self):
        """
        Render a Folium bubble map of departure delays.

        Returns:
            folium.Map: Folium map object with delay bubbles.
        """
        if len(self.merged) == 0:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        lats = [pt[0] for pt in self.merged['Geo_Point']]
        lons = [pt[1] for pt in self.merged['Geo_Point']]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(location=(center_lat, center_lon), zoom_start=7, tiles="cartodb positron")

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            radius = min(max(delay_min / 10, 3), 15)

            popup = f"{row['Name_FR']}<br>Total Departure Delay: {round(delay_min, 1)} min"

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color="crimson",
                fill=True,
                fill_color="orange",
                fill_opacity=0.7,
                popup=popup,
                tooltip=popup
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m





import pandas as pd
import plotly.express as px
import warnings

class DelayHeatmap:
    """
    Generates heatmaps of delays (arrival or departure) by station and hour using Plotly.
    Supports optional filtering by station group (e.g., Brussels, Antwerp, Others).
    """
    def __init__(self, delay_data_path: str, top_n: int = 10):
        self.delay_data_path = delay_data_path
        self.df = pd.read_csv(delay_data_path)
        self.top_n = top_n
        self.top_stations = None
        self.pivot_table = None
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()

    def _get_top_arrival_stations(self):
        df = self.df.copy()
        df["Delay at arrival"] = pd.to_numeric(df["Delay at arrival"], errors="coerce")
        df = df[df["Delay at arrival"] > 0]

        top_stations = (
            df.groupby("Stopping place (FR)")["Delay at arrival"]
            .sum()
            .div(60)
            .sort_values(ascending=False)
            .head(self.top_n)
            .index
            .tolist()
        )
        self.top_stations = top_stations
        return top_stations

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors='coerce')
        return parsed

    def _prepare_heatmap_data(self, delay_type="departure", station_filter=None):
        if self.top_stations is None:
            self._get_top_arrival_stations()

        delay_col = "Delay at departure" if delay_type == "departure" else "Delay at arrival"
        time_col = "Actual departure time" if delay_type == "departure" else "Actual arrival time"

        df = self.df.copy()
        df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce")
        df[time_col] = self._parse_datetime_column(df, time_col)
        df["Hour"] = df[time_col].dt.hour
        df = df.dropna(subset=["Hour", delay_col])
        df = df[df[delay_col] > 0]

        # Filter by station group if specified
        if station_filter is not None:
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        grouped = (
            df.groupby(["Stopping place (FR)", "Hour"])[delay_col]
            .sum()
            .div(60)
            .reset_index()
        )

        pivot = grouped.pivot(index="Stopping place (FR)", columns="Hour", values=delay_col).fillna(0)
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

        self.pivot_table = pivot
        return pivot
    def render_heatmap(self, delay_type="departure", station_filter=None):
        if delay_type not in ["departure", "arrival"]:
            raise ValueError("delay_type must be 'departure' or 'arrival'")

        title = f"{delay_type.title()} Delay Heatmap (Filtered Stations)"
        self._prepare_heatmap_data(delay_type=delay_type, station_filter=station_filter)

        fig = px.imshow(
            self.pivot_table,
            labels=dict(x="Hour of Day", y="Station", color="Total Delay (min)"),
            x=self.pivot_table.columns,
            y=self.pivot_table.index,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title=title
        )

        fig.update_layout(margin=dict(t=50, l=100, r=20, b=50))
        fig.update_xaxes(type='category')
        return fig