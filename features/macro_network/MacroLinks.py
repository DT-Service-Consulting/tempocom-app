from altair import Undefined
from tempocom.services import DBConnector
import pandas as pd
import ast
from features.operational_points.OperationalPoints import OperationalPoints
import networkx as nx
import itertools as it
import folium

class MacroLinks:
    sql_query = """
    SELECT
    Departure_PTCAR_ID as DEPARTURE_ID,
    Arrival_PTCAR_ID as ARRIVAL_ID,
    Distance as DISTANCE,
    Geo_Shape as GEO_SHAPE
    FROM macro_network
    """
    
    def __init__(self,dbc:DBConnector,lang = "EN"):
        self.df = pd.DataFrame(dbc.query(self.sql_query))
        self.df['DISABLED'] = False
        self.df['GEO_SHAPE'] = self.df['GEO_SHAPE'].apply(ast.literal_eval)
        self.operational_points = OperationalPoints(dbc, lang)
    
    def update_operational_points_with_links_info(self, operational_points:OperationalPoints):
        # Get all macro station IDs
        macro_station_ids = set(self.df['DEPARTURE_ID']).union(set(self.df['ARRIVAL_ID']))
        # Count links for each operational point
        op_ids = operational_points.df['ID']
        n_links = op_ids.apply(
            lambda op_id: (
                (self.df['DEPARTURE_ID'] == op_id).sum() + (self.df['ARRIVAL_ID'] == op_id).sum()
            )
        )
        isin_macro_network = op_ids.isin(list(macro_station_ids))
        operational_points.df['N_LINKS'] = n_links // 2
        operational_points.df['ISIN_MACRO_NETWORK'] = isin_macro_network
        return operational_points
    
    def get_path_coordinates(self, path:list[int] | None):
        if path is None:
            return None
        polyline = []
        for i in range(len(path) - 1):
            polyline.extend(self.df.loc[(self.df['DEPARTURE_ID'] == path[i]) & (self.df['ARRIVAL_ID'] == path[i+1]), 'GEO_SHAPE'].values[0])
        return polyline

    def get_path_distance(self, path:list[int] | None):
        if path is None:
            return None

        if len(path) <= 1:
            return 0.0
        total_distance = 0.0
        for i in range(len(path) - 1):
            segment_distance = self.df.loc[
                (self.df['DEPARTURE_ID'] == path[i]) & (self.df['ARRIVAL_ID'] == path[i+1]), 
                'DISTANCE'
            ].values
            if len(segment_distance) > 0:
                total_distance += float(segment_distance[0])
        return total_distance

    
    def get_macro_stations(self, oper_points:OperationalPoints):
        macro_stations_ids = set(self.df['DEPARTURE_ID']).union(set(self.df['ARRIVAL_ID']))
        macro_stations = oper_points.df[oper_points.df['ID'].isin(list(macro_stations_ids))]
        
        # Add a column 'n_links' to count the number of links for each station
        macro_stations = pd.DataFrame(macro_stations.copy())
        macro_stations.loc[:, 'N_LINKS'] = macro_stations['ID'].apply(
            lambda station_id: self.df[
                (self.df['DEPARTURE_ID'] == station_id) | (self.df['ARRIVAL_ID'] == station_id)
            ].shape[0] // 2  # Divide by 2 to account for round trips
        )
        
        return macro_stations

    def isin_macro_network(self, id) -> bool:
        return id in set(self.df['DEPARTURE_ID']).union(set(self.df['ARRIVAL_ID']))

    # ┌──────────────────────────────────────────────────────────┐
    # │         Macro Network Operations                         │
    # └──────────────────────────────────────────────────────────┘


    def close_links(self, closed_links: list[tuple]):
        # Initialize DISABLED column if it doesn't exist
        if 'DISABLED' not in self.df.columns:
            self.df['DISABLED'] = False
        
        # Mark links as disabled instead of removing them
        for departure_id, arrival_id in closed_links:
            mask = ((self.df['DEPARTURE_ID'] == departure_id) & (self.df['ARRIVAL_ID'] == arrival_id)) | \
                   ((self.df['DEPARTURE_ID'] == arrival_id) & (self.df['ARRIVAL_ID'] == departure_id))
            self.df.loc[mask, 'DISABLED'] = True
        return self.df

    # ┌──────────────────────────────────────────────────────────┐
    # │         Shortest Path Calculations                       │
    # └──────────────────────────────────────────────────────────┘

    def get_shortest_path(self, origin_id:int, extremity_id:int):
        self.graph = nx.from_pandas_edgelist(self.df.where(~self.df['DISABLED']),'DEPARTURE_ID', 'ARRIVAL_ID', edge_attr='DISTANCE')
        try:
            shortest_path = nx.shortest_path(self.graph, origin_id, extremity_id, weight='DISTANCE')
        except nx.NetworkXNoPath:
            return None
        return shortest_path
    
    def get_link_names(self):
        links = []
        for _, row in self.df.iterrows():
            if not row.get('DISABLED', False):
                dep_name = self.operational_points.get_name_by_id(row['DEPARTURE_ID'])
                arr_name = self.operational_points.get_name_by_id(row['ARRIVAL_ID'])
                links.append(f"{dep_name} ⇔ {arr_name}")
        return links

    def get_shortest_path_distance(self, origin_id:int, extremity_id:int):
        path = self.get_shortest_path(origin_id, extremity_id)
        return self.get_path_distance(path)
    def get_shortest_path_coordinates(self, origin_id:int, extremity_id:int):
        path = self.get_shortest_path(origin_id, extremity_id)
        return self.get_path_coordinates(path)

    def get_shortest_path_distance_coordinates(self, origin_id:int, extremity_id:int):
        path = self.get_shortest_path(origin_id, extremity_id)
        return self.get_path_coordinates(path), self.get_path_distance(path)
