import pandas as pd
import folium
import ast

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