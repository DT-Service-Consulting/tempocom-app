# Updated Delay_network.py with support for date filtering and severity coloring

import pandas as pd
import plotly.express as px
import warnings
import folium
import ast
import plotly.graph_objects as go

class DelayBubbleMap:
    def __init__(self, stations_path: str, delay_data_path: str):
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)

    def prepare_data(self, station_filter=None, date_filter=None):
        self.delays = self.delays[self.delays['Delay at arrival'] > 0].copy()

        self.delays['Actual arrival time'] = pd.to_datetime(self.delays['Actual arrival time'], errors='coerce')
        if date_filter:
            self.delays = self.delays[self.delays['Actual arrival time'].dt.date == date_filter]

        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at arrival']
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={'Delay at arrival': 'Total_Delay_Minutes'})
        ).nlargest(10, 'Total_Delay_Minutes')

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].str.title()

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

    def render_map(self):
        if self.merged.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        lats = [pt[0] for pt in self.merged['Geo_Point']]
        lons = [pt[1] for pt in self.merged['Geo_Point']]
        m = folium.Map(location=(sum(lats)/len(lats), sum(lons)/len(lons)), zoom_start=7, tiles="cartodb positron")

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            radius = min(max(delay_min / 10, 3), 15)
            if delay_min < 10:
                color = 'green'
            elif delay_min < 30:
                color = 'orange'
            else:
                color = 'red'

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{row['Name_FR']}<br>Arrival Delay: {round(delay_min, 1)} min"
            ).add_to(m)

        return m


class DelayBubbleMap2:
    def __init__(self, stations_path: str, delay_data_path: str):
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)

    def prepare_data1(self, station_filter=None, date_filter=None):
        self.delays = self.delays[self.delays['Delay at departure'] > 0].copy()

        self.delays['Actual departure time'] = pd.to_datetime(self.delays['Actual departure time'], errors='coerce')
        if date_filter:
            self.delays = self.delays[self.delays['Actual departure time'].dt.date == date_filter]

        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at departure']
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={'Delay at departure': 'Total_Delay_Minutes'})
        ).nlargest(10, 'Total_Delay_Minutes')

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].str.title()

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
        if self.merged.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        lats = [pt[0] for pt in self.merged['Geo_Point']]
        lons = [pt[1] for pt in self.merged['Geo_Point']]
        m = folium.Map(location=(sum(lats)/len(lats), sum(lons)/len(lons)), zoom_start=7, tiles="cartodb positron")

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            radius = min(max(delay_min / 10, 3), 15)
            if delay_min < 10:
                color = 'green'
            elif delay_min < 30:
                color = 'orange'
            else:
                color = 'red'

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{row['Name_FR']}<br>Departure Delay: {round(delay_min, 1)} min"
            ).add_to(m)

        return m


class DelayHeatmap:
    def __init__(self, delay_data_path):
        self.delay_data_path = delay_data_path

    def load_and_prepare(self, arrival=False, date_filter=None):
        df = pd.read_csv(self.delay_data_path)
        delay_col = "Delay at arrival" if arrival else "Delay at departure"
        time_col = "Actual arrival time" if arrival else "Actual departure time"

        df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce")
        df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        if date_filter:
            df = df[df[time_col].dt.date == date_filter]

        df["Hour"] = df[time_col].dt.hour
        df = df.dropna(subset=["Hour", delay_col])
        self.df = df

    def filter_and_prepare_heatmap(self, arrival=False, station_filter=None, top_n=10):
        df = self.df.copy()
        delay_col = "Delay at arrival" if arrival else "Delay at departure"

        if station_filter:
            station_filter = [s.title() for s in station_filter]
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        df["StopLabel"] = df["Stopping place (FR)"].str.title()

        top_stations = (
            df.groupby("StopLabel")[delay_col]
            .sum()
            .div(60)
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        df_top = df[df["StopLabel"].isin(top_stations)]

        heatmap_data = (
            df_top.groupby(["StopLabel", "Hour"])[delay_col]
            .sum()
            .reset_index()
        )

        pivot = heatmap_data.pivot(index="StopLabel", columns="Hour", values=delay_col).fillna(0)
        pivot["Total"] = pivot.sum(axis=1)
        self.pivot_table = pivot.sort_values("Total", ascending=False).drop(columns="Total")

    def render_heatmap(self, arrival=False):
        if self.pivot_table is None:
            raise ValueError("Run filter_and_prepare_heatmap() first.")

        title = "Arrival" if arrival else "Departure"
        return px.imshow(
            self.pivot_table,
            labels=dict(x="Hour", y="Station", color="Total Delay (min)"),
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title=f"{title} Delay Heatmap (Top 10 Stations)"
        )
