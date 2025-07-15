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
import plotly.graph_objects as go

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
    





class DelayLineChart:
    """
    Draws a line chart showing average departure and arrival delays over time,
    filtered by selected stations.
    """

    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.strip().str.title()
        self.df["Actual departure time"] = self._parse_datetime_column(self.df, "Actual departure time")
        self.df["Actual arrival time"] = self._parse_datetime_column(self.df, "Actual arrival time")
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
        return parsed

    def _aggregate_delays(self, station_filter=None, time_unit="hour"):
        df = self.df.copy()

        if station_filter:
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        if time_unit == "hour":
            df["Hour"] = df["Actual departure time"].dt.hour
        elif time_unit == "date":
            df["Hour"] = df["Actual departure time"].dt.date
        else:
            raise ValueError("time_unit must be 'hour' or 'date'")

        agg = df.groupby("Hour").agg({
            "Delay at departure": "mean",
            "Delay at arrival": "mean"
        }).dropna()

        return agg

    def render_line_chart(self, station_filter=None, time_unit="hour"):
        agg = self._aggregate_delays(station_filter=station_filter, time_unit=time_unit)

        fig = go.Figure()

        # Departure Delay in Gold
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at departure"],
            mode='lines+markers',
            name='Departure Delay (min)',
            line=dict(color='gold', width=2),
            marker=dict(color='gold')
        ))

        # Arrival Delay in Orange
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at arrival"],
            mode='lines+markers',
            name='Arrival Delay (min)',
            line=dict(color='red', width=2),
            marker=dict(color='red')
        ))

        fig.update_layout(
            title="📈 Average Delays per Hour (Selected Stations)",
            xaxis_title="Hour of Day" if time_unit == "hour" else "Date",
            yaxis_title="Delay (minutes)",
            xaxis=dict(
                type="category" if time_unit == "hour" else "linear",
                tickmode='linear',
                dtick=1,
                tick0=0,
                tickangle=0
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
            margin=dict(t=50, l=50, r=50, b=50),
            height=450
        )

        return fig





class DelayHourlyTotalLineChart:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_data()

    def _prepare_data(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.strip().str.title()
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Total Delay"] = self.df["Delay at departure"].fillna(0) + self.df["Delay at arrival"].fillna(0)

        # Combine time fields
        self.df["Hour"] = self.df["Actual departure time"].combine_first(self.df["Actual arrival time"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.df["Hour"] = pd.to_datetime(self.df["Hour"], errors="coerce").dt.hour

        self.df = self.df[self.df["Total Delay"] > 0]

    def plot(self, selected_stations=None, return_data=False):
        df = self.df.copy()
        df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str).str.title()
        df["Delay at departure"] = pd.to_numeric(df["Delay at departure"], errors="coerce")
        df["Delay at arrival"] = pd.to_numeric(df["Delay at arrival"], errors="coerce")

        # Combine delay columns
        df["Total Delay"] = df["Delay at departure"].fillna(0) + df["Delay at arrival"].fillna(0)
        df = df[df["Total Delay"] > 0]

        # Apply filter
        if selected_stations:
            df = df[df["Stopping place (FR)"].isin(selected_stations)]

        # Extract hour from combined time
        df["Hour"] = df["Actual departure time"].combine_first(df["Actual arrival time"])
        df["Hour"] = pd.to_datetime(df["Hour"], errors="coerce").dt.hour
        df = df.dropna(subset=["Hour"])

        # Group by station and hour
        grouped = (
            df.groupby(["Stopping place (FR)", "Hour"])["Total Delay"]
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={"Total Delay": "Total Delay (min)"})
        )

        if grouped.empty:
            return (None, grouped) if return_data else None

        # Plot
        import plotly.express as px
        fig = px.line(
            grouped,
            x="Hour",
            y="Total Delay (min)",
            color="Stopping place (FR)",
            markers=True,
            title="⏳ Hourly Total Delay by Station"
        )

        # Highlight max point per line
        max_points = grouped.loc[grouped.groupby("Stopping place (FR)")["Total Delay (min)"].idxmax()]
        fig.add_trace(px.scatter(
            max_points,
            x="Hour",
            y="Total Delay (min)",
            color_discrete_sequence=["red"],
            hover_name="Stopping place (FR)",
            hover_data=["Total Delay (min)"]
        ).data[0])

        fig.update_layout(
            xaxis=dict(title="Hour of Day", dtick=1),
            yaxis_title="Delay (minutes)",
            height=450,
            legend_title="Station"
        )

        return (fig, grouped) if return_data else fig






class DelayHourlyTotalLineChartByTrain:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_fields()

    def _prepare_fields(self):
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Total Delay"] = self.df["Delay at departure"].fillna(0) + self.df["Delay at arrival"].fillna(0)

        self.df["Hour"] = self.df["Actual departure time"].combine_first(self.df["Actual arrival time"])
        self.df["Hour"] = pd.to_datetime(self.df["Hour"], errors="coerce").dt.hour

    def plot(self, selected_trains=None, return_data=False):
        df = self.df.copy()
        df = df[df["Total Delay"] > 0]
        df = df.dropna(subset=["Hour"])

        if selected_trains:
            df = df[df["Train number"].isin(selected_trains)]
        
        # Sum delay per hour per train number
        grouped = (
            df.groupby(["Train number", "Hour"])["Total Delay"]
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={"Total Delay": "Total Delay (min)"})
        )

        # Ensure full 24-hour coverage
        all_hours = pd.DataFrame({'Hour': list(range(24))})
        full_data = []
        for train in grouped["Train number"].unique():
            train_data = grouped[grouped["Train number"] == train].merge(all_hours, on="Hour", how="right")
            train_data["Train number"] = train
            train_data["Total Delay (min)"] = train_data["Total Delay (min)"].fillna(0)
            full_data.append(train_data)
        
        grouped_full = pd.concat(full_data, ignore_index=True)

        if grouped_full.empty:
            return (None, grouped_full) if return_data else None

        import plotly.express as px
        fig = px.line(
            grouped_full,
            x="Hour",
            y="Total Delay (min)",
            color="Train number",
            markers=True,
            title="⏳ Hourly Total Delay by Train Number"
        )

        # Max points (after ensuring 24h coverage)
        max_points = grouped_full.loc[grouped_full.groupby("Train number")["Total Delay (min)"].idxmax()]
        fig.add_trace(px.scatter(
            max_points,
            x="Hour",
            y="Total Delay (min)",
            color_discrete_sequence=["red"],
            hover_name="Train number",
            hover_data=["Total Delay (min)"]
        ).data[0])

        fig.update_layout(
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1, range=[0, 23]),
            yaxis_title="Delay (minutes)",
            height=450,
            legend_title="Train Number"
        )

        return (fig, grouped_full) if return_data else fig



class DelayHourlyLinkTotalLineChart:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Actual departure time"] = self._parse_datetime_column(self.df, "Actual departure time")
        self.df["Actual arrival time"] = self._parse_datetime_column(self.df, "Actual arrival time")

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
        return parsed

    def _group_by_hour(self, df: pd.DataFrame, group_col: str):
        df["Total Delay"] = df["Delay at departure"].fillna(0) + df["Delay at arrival"].fillna(0)
        df = df[df["Total Delay"] > 0]
        df["Hour"] = df["Actual departure time"].combine_first(df["Actual arrival time"]).dt.hour
        grouped = df.groupby([group_col, "Hour"])["Total Delay"].sum().div(60).reset_index()
        return grouped

    def _plot_grouped(self, grouped: pd.DataFrame, group_col: str):
        if grouped.empty:
            return None
        fig = go.Figure()
        for group in grouped[group_col].unique():
            df_group = grouped[grouped[group_col] == group]
            max_val = df_group["Total Delay"].max()
            fig.add_trace(go.Scatter(
                x=df_group["Hour"],
                y=df_group["Total Delay"],
                mode="lines+markers+text",
                name=group,
                line=dict(width=2),
                text=[f"⬆️ {val:.1f}" if val == max_val else "" for val in df_group["Total Delay"]],
                textposition="top center"
            ))
        fig.update_layout(
            title=f"Total Delay by Hour per {group_col}",
            xaxis_title="Hour of Day",
            yaxis_title="Total Delay (min)",
            xaxis=dict(dtick=1),
            height=500,
            legend_title=group_col
        )
        return fig

    def plot(self, selected_stations=None, return_data=False):
        df = self.df.copy()
        if selected_stations:
            df = df[df["Stopping place (FR)"].isin(selected_stations)]
        grouped = self._group_by_hour(df, "Stopping place (FR)")
        fig = self._plot_grouped(grouped, "Stopping place (FR)")
        return (fig, grouped) if return_data else fig

    def plot_by_train_number(self, selected_trains=None, return_data=False):
        df = self.df.copy()
        if selected_trains:
            df = df[df["Train number"].isin(selected_trains)]
        grouped = self._group_by_hour(df, "Train number")
        fig = self._plot_grouped(grouped, "Train number")
        return (fig, grouped) if return_data else fig

    def plot_by_relation_direction(self, selected_relations=None, return_data=False):
        df = self.df.copy()
        if selected_relations:
            df = df[df["Relation direction"].isin(selected_relations)]
        grouped = self._group_by_hour(df, "Relation direction")
        fig = self._plot_grouped(grouped, "Relation direction")
        return (fig, grouped) if return_data else fig




class DelayNetwork:
    def __init__(self, delay_data_path):
        self.delay_data_path = delay_data_path
        self.df = None
        self.df_processed = None

    def load_data(self):
        # Load raw delay data CSV
        self.df = pd.read_csv(self.delay_data_path)
        # Convert date column to datetime
        self.df['Date of departure'] = pd.to_datetime(self.df['Date of departure'])

    def preprocess_data(self):
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        def get_cause(row):
            if row['Delay at arrival'] / 60 > 1:
                return 'Late Arrival'
            elif row['Delay at departure'] / 60 > 1:
                return 'Late Departure'
            else:
                return 'On Time'

        def get_severity(delay):
            delay_min = delay / 60
            if delay_min <= 5:
                return 'Low'
            elif delay_min <= 15:
                return 'Medium'
            else:
                return 'High'

        df['Cause'] = df.apply(get_cause, axis=1)
        df['Delay Severity'] = df['Delay at departure'].apply(get_severity)

        self.df_processed = df

    def filter_stations(self, focus_stations):
        if self.df_processed is None:
            self.preprocess_data()

        df_filtered = self.df_processed[self.df_processed['Stopping place (FR)'].isin(focus_stations)].copy()

        # Normalize station names
        df_filtered['Stopping place (FR)'] = (
            df_filtered['Stopping place (FR)']
            .str.replace("Brussel", "Brussels", regex=False)
            .str.replace("-", " ", regex=False)
        )

        return df_filtered

    def plot_sankey(self, focus_stations):
        """
        Generates a Sankey diagram figure visualizing delay causes, stations, and delay severity.
        """
        df = self.filter_stations(focus_stations)
        causes = sorted(df['Cause'].unique())
        stations = sorted(df['Stopping place (FR)'].unique())
        severities = ['Low', 'Medium', 'High']

        raw_labels = causes + stations + severities
        total_rows = len(df)

        node_counts = pd.concat([
            df['Cause'],
            df['Stopping place (FR)'],
            df['Delay Severity']
        ]).value_counts()

        label_map = {label: idx for idx, label in enumerate(raw_labels)}
        labels = [
            f"{label} ({(node_counts[label]/total_rows)*100:.1f}%)" if label in node_counts else label
            for label in raw_labels
        ]

        node_colors = (
            ['#e74c3c', '#f39c12', '#95a5a6'] +      # Causes: red, orange, gray
            ['#3498db'] * len(stations) +            # Stations: blue
            ['#2ecc71', '#f1c40f', '#e67e22']        # Severities: green, yellow, orange
        )

        cause_colors = {'Late Arrival': '#e74c3c', 'Late Departure': "#f39c12", 'On Time': '#95a5a6'}
        severity_colors = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e67e22'}

        source, target, value, link_colors, link_labels = [], [], [], [], []

        # Cause → Station links
        for cause in causes:
            for station in stations:
                count = df[(df['Cause'] == cause) & (df['Stopping place (FR)'] == station)].shape[0]
                if count > 5:
                    s = label_map[cause]
                    t = label_map[station]
                    p = count / total_rows * 100
                    source.append(s)
                    target.append(t)
                    value.append(count)
                    link_colors.append(cause_colors[cause])
                    link_labels.append(f"{cause} → {station}: {count} ({p:.1f}%)")

        # Station → Severity links
        for station in stations:
            for severity in severities:
                count = df[(df['Stopping place (FR)'] == station) & (df['Delay Severity'] == severity)].shape[0]
                if count > 5:
                    s = label_map[station]
                    t = label_map[severity]
                    p = count / total_rows * 100
                    source.append(s)
                    target.append(t)
                    value.append(count)
                    link_colors.append(severity_colors[severity])
                    link_labels.append(f"{station} → {severity}: {count} ({p:.1f}%)")

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=15,
                thickness=20,
                color=node_colors,
                line=dict(color='black', width=0.5)
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                label=link_labels,
            )
        )])

        fig.update_layout(
            title_text="🚦 Train Delay Flow: Cause → Station → Severity (Node & Link Percentages)",
            font_size=12,
            height=720,
            margin=dict(l=10, r=10, t=50, b=10)
        )

    

        return fig
    


# Delay_network.py (updated)

# ... [all previous imports and classes unchanged] ...

class DelayCausalityGraph:
    """
    Identifies and visualizes the longest chain of causally connected train delays.
    Supports interactive selection of a day and plots the causal graph with directional arrows.

    Dependencies:
    - pandas
    - networkx
    - plotly
    - ipywidgets (for Jupyter use)
    """
    def __init__(self, delay_data_path):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_fields()

    def _prepare_fields(self):
        self.df['Date of departure'] = pd.to_datetime(self.df['Date of departure'], format='%d%b%Y', errors='coerce')
        self.df['Actual departure time'] = pd.to_datetime(self.df['Actual departure time'], format='%H:%M:%S', errors='coerce')
        self.df['Planned departure time'] = pd.to_datetime(self.df['Planned departure time'], format='%H:%M:%S', errors='coerce')

    def _find_longest_path(self, G):
        longest_path = []

        def dfs(node, path):
            nonlocal longest_path
            path.append(node)
            if len(path) > len(longest_path):
                longest_path = path.copy()
            for neighbor in G.successors(node):
                if neighbor not in path:
                    dfs(neighbor, path)
            path.pop()

        for node in G.nodes():
            dfs(node, [])

        return longest_path

    def plot_longest_chain(self, selected_date):
        import plotly.graph_objects as go
        import networkx as nx
        import pandas as pd

        selected_date = pd.to_datetime(selected_date).date()
        df_day = self.df[self.df['Date of departure'].dt.date == selected_date].copy()
        df_day = df_day[df_day['Delay at departure'] >= 5]
        if df_day.empty:
            print(f"No delay data for {selected_date}")
            return

        G = nx.DiGraph()
        added_trains = set()
        TIME_WINDOW = pd.Timedelta(minutes=15)
        MAX_NODES = 20

        for _, group in df_day.groupby('Stopping place (FR)'):
            trains = group.sort_values('Actual departure time').to_dict('records')
            for i, t1 in enumerate(trains):
                id1 = f"Train {t1['Train number']}"
                if id1 not in added_trains and len(added_trains) < MAX_NODES:
                    G.add_node(id1)
                    added_trains.add(id1)
                for j in range(i + 1, len(trains)):
                    t2 = trains[j]
                    id2 = f"Train {t2['Train number']}"
                    if (
                        t2['Planned departure time'] > t1['Actual departure time']
                        and t2['Planned departure time'] - t1['Actual departure time'] <= TIME_WINDOW
                        and t2['Delay at departure'] >= 5
                    ):
                        if id2 not in added_trains and len(added_trains) < MAX_NODES:
                            G.add_node(id2)
                            added_trains.add(id2)
                        G.add_edge(id1, id2)

        longest_chain = self._find_longest_path(G)

        if not longest_chain:
            print(f"No causality chain found for {selected_date}")
            return

        # Layout and visualization
        pos = {node: (i * 3, 0) for i, node in enumerate(longest_chain)}
        edge_x, edge_y, node_x, node_y = [], [], [], []
        annotations = []

        for u, v in zip(longest_chain[:-1], longest_chain[1:]):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            annotations.append(dict(ax=x0, ay=y0, x=x1, y=y1, arrowhead=3, showarrow=True,
                                    arrowwidth=2, arrowcolor="crimson"))

        for node in longest_chain:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='crimson', width=2)))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=longest_chain,
                                 marker=dict(size=16, color='tomato'), textposition='top center'))

        fig.update_layout(
            title=f"Train Delay Causality Chain on {selected_date}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        return fig
