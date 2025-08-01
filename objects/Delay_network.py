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
"""# Updated Delay_network.py with support for date filtering and severity coloring

import pandas as pd
import plotly.express as px
import warnings
import folium
import ast
import plotly.graph_objects as go

import pandas as pd
import folium
import ast

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
        )

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.strip().str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].astype(str).str.strip().str.title()

        if station_filter:
            station_filter = [s.strip().title() for s in station_filter]
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

        delays = self.merged['Total_Delay_Minutes']
        min_delay, max_delay = delays.min(), delays.max()

        def delay_to_color(delay):
            norm = (delay - min_delay) / (max_delay - min_delay + 1e-6)
            r = int(255 * norm)
            g = int(255 * (1 - norm))
            return f"#{r:02x}{g:02x}00"

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            norm = (delay_min - min_delay) / (max_delay - min_delay + 1e-6)
            radius = 3 + norm * 12

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color=delay_to_color(delay_min),
                fill=True,
                fill_color=delay_to_color(delay_min),
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

        self.delays['Actual departure time'] = pd.to_datetime(
            self.delays['Actual departure time'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
        )
        if date_filter:
            self.delays = self.delays[self.delays['Actual departure time'].dt.date == date_filter]

        self.delay_summary = (
            self.delays.groupby('Stopping place (FR)')['Delay at departure']
            .sum().div(60).reset_index()
            .rename(columns={'Delay at departure': 'Total_Delay_Minutes'})
        )

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.strip().str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].astype(str).str.strip().str.title()

        if station_filter:
            station_filter = [s.strip().title() for s in station_filter]
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

        delays = self.merged['Total_Delay_Minutes']
        min_delay, max_delay = delays.min(), delays.max()

        def delay_to_color(delay):
            norm = (delay - min_delay) / (max_delay - min_delay + 1e-6)
            r = int(255 * norm)
            g = int(255 * (1 - norm))
            return f"#{r:02x}{g:02x}00"

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            norm = (delay_min - min_delay) / (max_delay - min_delay + 1e-6)
            radius = 3 + norm * 12

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color=delay_to_color(delay_min),
                fill=True,
                fill_color=delay_to_color(delay_min),
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
            df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")

            if date_filter:
                # Ensure date_filter is a datetime.date
                if isinstance(date_filter, pd.Timestamp):
                    date_filter = date_filter.date()
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

import pandas as pd
import folium
import ast

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
        )

        self.delay_summary['Stopping place (FR)'] = self.delay_summary['Stopping place (FR)'].str.strip().str.title()
        self.stations['Name_FR'] = self.stations['Name_FR'].astype(str).str.strip().str.title()

        if station_filter:
            station_filter = [s.strip().title() for s in station_filter]
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

        delays = self.merged['Total_Delay_Minutes']
        min_delay, max_delay = delays.min(), delays.max()

        def delay_to_color(delay):
            norm = (delay - min_delay) / (max_delay - min_delay + 1e-6)
            r = int(255 * norm)
            g = int(255 * (1 - norm))
            return f"#{r:02x}{g:02x}00"

        for _, row in self.merged.iterrows():
            delay_min = row['Total_Delay_Minutes']
            norm = (delay_min - min_delay) / (max_delay - min_delay + 1e-6)
            radius = 3 + norm * 12

            folium.CircleMarker(
                location=row['Geo_Point'],
                radius=radius,
                color=delay_to_color(delay_min),
                fill=True,
                fill_color=delay_to_color(delay_min),
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
