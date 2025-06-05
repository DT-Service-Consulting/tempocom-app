import pandas as pd
import folium
import ast
import matplotlib.pyplot as plt
import seaborn as sns

class DelayBubbleMap:
    def __init__(self, stations_path: str, delay_data_path: str):
        self.stations = pd.read_csv(stations_path)
        self.delays = pd.read_csv(delay_data_path)

        self.prepare_data()

    def prepare_data(self, station_filter=None):
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
        self.delay_summary = self.delay_summary.nlargest(20, 'Total_Delay_Minutes')
        


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
        def __init__(self, stations_path: str, delay_data_path: str):
            self.stations = pd.read_csv(stations_path)
            self.delays = pd.read_csv(delay_data_path)

            self.prepare_data1()



        def prepare_data1(self, station_filter=None):
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
            self.delay_summary = self.delay_summary.nlargest(20, 'Total_Delay_Minutes')

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
    def __init__(self, delay_data_path):
        self.delay_data_path = delay_data_path
        self.df = None
        self.pivot_table = None

    def load_and_prepare(self):
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
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        df_top = df[df["StopLabel"].isin(top_stations)]

        heatmap_data = (
            df_top.groupby(["StopLabel", "Hour"])["Delay at departure"]
            .mean()
            .reset_index()
        )

        pivot = heatmap_data.pivot(index="StopLabel", columns="Hour", values="Delay at departure").fillna(0)
        pivot["Avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("Avg", ascending=False).drop(columns="Avg")

        self.pivot_table = pivot

    def render_heatmap(self, title="Departure Delay Heatmap (Top 10 Stations)", figsize=(12, 6)):
        if self.pivot_table is None:
            raise ValueError("Run filter_and_prepare_heatmap() first.")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self.pivot_table, cmap="YlOrRd", linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Station")
        plt.xticks(rotation=45)
        plt.tight_layout()
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
        df = self.df.copy()

        if station_filter:
            station_filter = [s.title() for s in station_filter]
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        top_stations = (
            df.groupby("StopLabel")["Delay at arrival"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        df_top = df[df["StopLabel"].isin(top_stations)]

        heatmap_data = (
            df_top.groupby(["StopLabel", "Hour"])["Delay at arrival"]
            .mean()
            .reset_index()
        )

        pivot = heatmap_data.pivot(index="StopLabel", columns="Hour", values="Delay at arrival").fillna(0)
        pivot["Avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("Avg", ascending=False).drop(columns="Avg")

        self.pivot_table = pivot

    def render_heatmap1(self, title="Arrival Delay Heatmap (Top 10 Stations)", figsize=(12, 6)):
        if self.pivot_table is None:
            raise ValueError("Run filter_and_prepare_heatmap1() first.")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self.pivot_table, cmap="YlOrRd", linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Station")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
