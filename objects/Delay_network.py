"""
Delay Network Visualization Module

This module provides classes for visualizing train delay data using various interactive and static methods:

Classes
-------
- DelayBubbleMap:
    Visualizes total arrival delays per station as a bubble map using Folium.
- DelayBubbleMap2:
    Visualizes total departure delays per station as a bubble map using Folium.
- DelayHeatmap:
    Generates heatmaps of delays (arrival or departure) by station and hour using Plotly.

Each class provides methods to load, process, and visualize delay data, supporting filtering and aggregation for flexible analysis.

Dependencies
------------
- pandas
- folium
- ast
- matplotlib
- seaborn
- plotly

Typical Usage
-------------
Instantiate the desired class with the appropriate CSV paths, call the data preparation method, and then render the visualization.

Example:
    bubble_map = DelayBubbleMap(stations_path, delay_data_path)
    folium_map = bubble_map.render_map()"""



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
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)

        self.prepare_data()
        

    def prepare_data(self, station_filter=None):
        """
        Initialize DelayBubbleMap with station and delay data CSV paths.

        Args:
            stations_path (str): Path to stations CSV file.
            delay_data_path (str): Path to delay data CSV file.
        """
        # Filter out negative delays
        self.delays = self.delays[self.delays['Delay at arrival'] > 0]

        # Sum delay in **minutes** per station
        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at arrival']
            .sum()
            .div(60)  # Convert from seconds to minutes
            .reset_index()
            .rename(columns={'Delay at arrival': 'Total_Delay_Minutes'})
        )
        self.delay_summary = self.delay_summary.nlargest(10, 'Total_Delay_Minutes')
        


        # Normalize station names to title case for joining
        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].str.title()
        if station_filter:
            station_filter = [s.title() for s in station_filter]
            self.delay_summary = self.delay_summary[self.delay_summary['Stopping place (FR)'].isin(station_filter)]

        # Merge delay data with station info
        self.merged = self.delay_summary.merge(
            self.stations[['Name_FR', 'Geo_Point']],
            left_on='Stopping place (FR)',
            right_on='Name_FR',
            how='inner'
        )

        # Parse coordinates
        self.merged['Geo_Point'] = self.merged['Geo_Point'].apply(ast.literal_eval)

    def render_map(self):
        """Render a Folium bubble map of arrival delays.

        Returns:
            folium.Map: Folium map object with delay bubbles."""
        
        if len(self.merged) == 0:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        # Auto-center
        lats = [pt[0] for pt in self.merged['Geo_Point']]
        lons = [pt[1] for pt in self.merged['Geo_Point']]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(location=(center_lat, center_lon), zoom_start=7, tiles="cartodb positron")

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            radius = min(max(delay_min / 10, 3), 15)  # scale bubble size

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
        """ Visualizes total departure delays per station as a bubble map using Folium."""
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
                Prepare and merge departure delay and station data for visualization.

                Args:
                    station_filter (list, optional): List of station names to filter. Defaults to None.
            """
            # Filter out negative delays
            self.delays = self.delays[self.delays['Delay at departure'] > 0]

            # Sum delay in minutes per station
            self.delay_summary = (
                self.delays.groupby('Stopping place (FR)')['Delay at departure']
                .sum()
                .div(60)
                .reset_index()
                .rename(columns={'Delay at departure': 'Total_Delay_Minutes'})
            )

            # Title case for merging
            self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.title()
            self.stations['Name_FR'] = self.stations['Name_FR'].str.title()

            # Top 10 stations
            self.delay_summary = self.delay_summary.nlargest(10, 'Total_Delay_Minutes')

            # ✅ APPLY STATION FILTER HERE
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

            # Parse coordinates
            self.merged['Geo_Point'] = self.merged['Geo_Point'].apply(ast.literal_eval)
                    
        def render_map1(self):
            """Render a Folium bubble map of departure delays.
            Returns:
            folium.Map: Folium map object with delay bubbles.
        """
            if len(self.merged) == 0:
                return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

            # Auto-center map
            lats = [pt[0] for pt in self.merged['Geo_Point']]
            lons = [pt[1] for pt in self.merged['Geo_Point']]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            m = folium.Map(location=(center_lat, center_lon), zoom_start=7, tiles="cartodb positron")

            for _, row in self.merged.iterrows():
                delay_min = row['Total_Delay_Minutes']
                radius = min(max(delay_min / 10, 3), 15)  # scale bubble size

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







class DelayHeatmap:
    """
    Generates heatmaps of delays (arrival or departure) by station and hour.
    """
    def __init__(self, delay_data_path):
        """
        Initialize DelayHeatmap with delay data CSV path.

        Args:
            delay_data_path (str): Path to delay data CSV file.
        """
        self.delay_data_path = delay_data_path
        self.df = None
        self.pivot_table = None

    def load_and_prepare(self):
        """
        Load and prepare data for departure delay heatmap.
        """
        df = pd.read_csv(self.delay_data_path)

        df["Delay at departure"] = pd.to_numeric(df["Delay at departure"], errors="coerce")
        df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str)
        df["Start Station (FR)"] = df["Start Station (FR)"].astype(str)
        df["End Station (FR)"] = df["End Station (FR)"].astype(str)

        df["Actual departure time"] = pd.to_datetime(df["Actual departure time"], errors="coerce")
        df["Hour"] = df["Actual departure time"].dt.hour

        df["StopLabel"] = (
            df["Stopping place (FR)"].str.title() +
            " (Start: " + df["Start Station (FR)"].str.title() +
            " → End: " + df["End Station (FR)"].str.title() + ")"
        )

        df = df.dropna(subset=["Hour", "StopLabel", "Delay at departure"])
        self.df = df

    def filter_and_prepare_heatmap(self, station_filter=None, top_n=10):
        df = self.df.copy()

        if station_filter:
            station_filter = [s.title() for s in station_filter]
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        top_stations = (
        df.groupby("StopLabel")["Delay at departure"]
        .sum()
        .div(60)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

        df_top = df[df["StopLabel"].isin(top_stations)]

        heatmap_data = (
            df_top.groupby(["StopLabel", "Hour"])["Delay at departure"]
            .sum()
            .reset_index()
        )

        pivot = heatmap_data.pivot(index="StopLabel", columns="Hour", values="Delay at departure").fillna(0)
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

        self.pivot_table = pivot

    def render_heatmap(self, title="Departure Delay Heatmap (Top 10 Stations by Total Delay)"):
        if self.pivot_table is None:
            raise ValueError("Run filter_and_prepare_heatmap() first.")

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

    def load_and_prepare1(self):
        df = pd.read_csv(self.delay_data_path)

        df["Delay at arrival"] = pd.to_numeric(df["Delay at arrival"], errors="coerce")
        df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str)
        df["Start Station (FR)"] = df["Start Station (FR)"].astype(str)
        df["End Station (FR)"] = df["End Station (FR)"].astype(str)

        df["Actual arrival time"] = pd.to_datetime(df["Actual arrival time"], errors="coerce")
        df["Hour"] = df["Actual arrival time"].dt.hour

        df["StopLabel"] = (
            df["Stopping place (FR)"].str.title() +
            " (Start: " + df["Start Station (FR)"].str.title() +
            " → End: " + df["End Station (FR)"].str.title() + ")"
        )

        df = df.dropna(subset=["Hour", "StopLabel", "Delay at arrival"])
        self.df = df

    def filter_and_prepare_heatmap1(self, station_filter=None, top_n=10):
        """
        Filter and aggregate data for departure delay heatmap.

        Args:
            station_filter (list, optional): List of station names to filter. Defaults to None.
            top_n (int, optional): Number of top stations to include. Defaults to 10.
        """
        df = self.df.copy()

        if station_filter:
            station_filter = [s.title() for s in station_filter]
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        top_stations = (
            df.groupby("StopLabel")["Delay at arrival"]
            .sum()
            .div(60)  # Convert from seconds to minutes
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        df_top = df[df["StopLabel"].isin(top_stations)]

        heatmap_data = (
            df_top.groupby(["StopLabel", "Hour"])["Delay at arrival"]
            .sum()
            .reset_index()
        )

        pivot = heatmap_data.pivot(index="StopLabel", columns="Hour", values="Delay at arrival").fillna(0)
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

        self.pivot_table = pivot

    def render_heatmap1(self, title="Arrival Delay Heatmap (Top 10 Stations by Total Delay)"):
        """
        Render a Plotly heatmap for departure delays.

        Args:
            title (str, optional): Title for the heatmap. Defaults to "Departure Delay Heatmap (Top 10 Stations by Total Delay)".

        Returns:
            plotly.graph_objs._figure.Figure: Plotly heatmap figure.
        """
        if self.pivot_table is None:
            raise ValueError("Run filter_and_prepare_heatmap1() first.")

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
