from utils import get_mart
import folium
import pandas as pd
import ast
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import os
from dotenv import load_dotenv
from modules.DBConnector import DBConnector
from pyspark.sql import SparkSession

class MacroNetwork:
    """
    A class for managing and visualizing railway macro network data.
    
    This class handles railway network data including stations, links, and provides
    methods for network analysis, shortest path calculations, and map rendering.
    
    Attributes:
        links (DataFrame): Network links data with geometry and distance information
        stations (DataFrame): Station data with location and classification information
    """

    def __init__(self,path_to_mart:str=os.getenv('MART_RELATIVE_PATH')):
        """
        Initialize the MacroNetwork object with data from the mart directory.
        
        Args:
            path_to_mart (str): Path to the mart directory containing data files.
                Defaults to MART_RELATIVE_PATH environment variable.
                
        Note:
            Loads network_graph.csv and stations.csv, processes geometry data,
            and computes the number of links per station.
        """
        # extracting
        dbc = DBConnector()
        mn_query = """
        SELECT 
    mn.*,

    -- Infos sur le point de départ
    opd."Complete_name_in_French" AS Departure_Name_FR,

    -- Infos sur le point d’arrivée
    opa."Complete_name_in_French" AS Arrival_Name_FR,

    -- Nouvelle colonne disabled
    0 AS disabled

    FROM 
        macro_network mn

    -- Jointure pour les points de départ
    INNER JOIN 
        operational_points opd
        ON mn.Departure_PTCAR_ID = opd.PTCAR_ID

    -- Jointure pour les points d’arrivée
    INNER JOIN 
        operational_points opa
        ON mn.Arrival_PTCAR_ID = opa.PTCAR_ID"""


        self.links = pd.DataFrame(dbc.query(mn_query))
        print(self.links.head())
        self.stations = pd.DataFrame(dbc.query("SELECT Geo_Point, PTCAR_ID, Complete_name_in_French as Name_FR, Classification_EN as Classification_FR FROM operational_points where isin_macro_network = 1"))
        print(self.stations.head())
        #self.available_links = self.links[self.links['disabled'] == 0]
        
        #processing
        self.compute_number_of_links()
        self.links['Geo_Shape'] = self.links['Geo_Shape'].apply(lambda x: ast.literal_eval(x))
        self.links['Distance'] = self.links['Distance'].apply(lambda x: round(float(x), 1))
        self.stations['Geo_Point'] = self.stations['Geo_Point'].apply(lambda x: ast.literal_eval(x))

    def get_station_by_id(self, ptcarid):
        """
        Get station information by PTCAR_ID.
        
        Args:
            ptcarid: The PTCAR_ID of the station
            
        Returns:
            Series: Station data row
        """
        return self.stations[self.stations['PTCAR_ID'] == ptcarid].iloc[0]
    
    def get_link_by_ids(self, id1, id2):
        """
        Get link information between two stations by their PTCAR_IDs.
        
        Args:
            id1: PTCAR_ID of the first station
            id2: PTCAR_ID of the second station
            
        Returns:
            Series: Link data row (works in both directions)
        """
        return self.links[(self.links['Departure_PTCAR_ID'] == id1) & (self.links['Arrival_PTCAR_ID'] == id2)
                          | (self.links['Departure_PTCAR_ID'] == id2) & (self.links['Arrival_PTCAR_ID'] == id1)].iloc[0]
    
    def get_open_links(self):
        """
        Get list of open (non-disabled) links in the network.
        
        Returns:
            list: List of connection strings in format "Station1 ⇔ Station2"
        """
        open_links = self.links[self.links['disabled'] == 0]
        connections = []
        for _, link in open_links.iterrows():
            conn = f"{link['Departure_Name_FR']} ⇔ {link['Arrival_Name_FR']}"
            reverse_conn = f"{link['Arrival_Name_FR']} ⇔ {link['Departure_Name_FR']}"
            if conn not in connections and reverse_conn not in connections:
                connections.append(conn)
        return connections
    
    def render_link(self,link,color="grey"):
        """
        Render a network link as a Folium PolyLine.
        
        Args:
            link (Series): Link data containing geometry and station information
            color (str, optional): Color of the line. Defaults to "grey".
            
        Returns:
            folium.PolyLine: Folium PolyLine object for the link
        """
        text = f"{link['Departure_Name_FR']} ⇔ {link['Arrival_Name_FR']} ({link['Distance']} km)"
        return folium.PolyLine(
            locations=link['Geo_Shape'],
            color=color,
            weight=3,
            popup=text,
            tooltip=text
        )
    
    def render_station(self,station,color="white",fill_color="black"):
        """
        Render a station as a Folium CircleMarker.
        
        Args:
            station (Series): Station data containing location and link information
            color (str, optional): Border color of the marker. Defaults to "white".
            fill_color (str, optional): Fill color of the marker. Defaults to "black".
            
        Returns:
            folium.CircleMarker: Folium CircleMarker object for the station
        """
        text = f"{station['Name_FR']} ({station['n_links']} links) - {station['Classification_FR']}"
        # Rayon : 1 si <=2 connexions, sinon 3
        radius = 1 if station['n_links'] <= 2 else 3
        return folium.CircleMarker(
            location=station['Geo_Point'],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            tooltip=text,
            popup=text
        )
    
    def compute_number_of_links(self):
        """
        Compute the number of links connected to each station.
        
        Note:
            Updates the 'n_links' column in the stations DataFrame.
            The count is divided by 2 since each link connects two stations.
        """
        self.stations['n_links'] = 0
        
        for _, link in self.links.iterrows():
            dep_id = link['Departure_PTCAR_ID']
            arr_id = link['Arrival_PTCAR_ID']
            
            self.stations.loc[self.stations['PTCAR_ID'] == dep_id, 'n_links'] += 1
            self.stations.loc[self.stations['PTCAR_ID'] == arr_id, 'n_links'] += 1
            
        self.stations['n_links'] = self.stations['n_links'] / 2
        self.stations['n_links'] = self.stations['n_links'].astype(int)

    def render_macro_network(self,m):
        """
        Render the complete macro network on a Folium map.
        
        Args:
            m (folium.Map): The Folium map object to add the network to
            
        Returns:
            folium.Map: Updated map with network layers added
        """
        links_layer = folium.FeatureGroup(name='Links')
        stations_layer = folium.FeatureGroup(name='Stations')
        done_links = []

        #render links
        for _, link in self.links.iterrows():
            link_key = tuple(sorted([link['Departure_Name_FR'], link['Arrival_Name_FR']]))
            if link_key in done_links:
                continue
            done_links.append(link_key)

            self.render_link(link).add_to(links_layer)

        #render stations
        for _, info in self.stations.iterrows():
            self.render_station(info).add_to(stations_layer)

        links_layer.add_to(m)
        stations_layer.add_to(m)

        return m
    
    def close_links(self, closed_links: list[tuple], m=None):
        """
        Close specified links in the network and optionally render them on the map.
        
        Args:
            closed_links (list[tuple]): List of tuples containing station name pairs to close
            m (folium.Map, optional): Folium map to render closed links on. Defaults to None.
            
        Returns:
            folium.Map or None: Updated map with closed links rendered, or None if no map provided
        """
        edited_network = folium.FeatureGroup(name='Closed Links')

        #disable both ways
        for link in closed_links:
            mask_way1 = (self.links['Departure_Name_FR'] == link[0]) & (self.links['Arrival_Name_FR'] == link[1])
            self.links.loc[mask_way1, 'disabled'] = 1
            mask_way2 = (self.links['Departure_Name_FR'] == link[1]) & (self.links['Arrival_Name_FR'] == link[0])
            self.links.loc[mask_way2, 'disabled'] = 1
        if not m:
            return
        #render edited network  
        for _, link in self.links.loc[self.links['disabled'] == 1].iterrows():
            self.render_link(link,color="red").add_to(edited_network)
        edited_network.add_to(m)
        self.available_links = self.links[self.links['disabled'] == 0]
        return m
            

    def get_shortest_path(self, start_station, end_station):
        """
        Calculate the shortest path between two stations using Floyd-Warshall algorithm.
        
        Args:
            start_station (str): Name of the starting station
            end_station (str): Name of the destination station
            
        Returns:
            tuple: (path, total_distance) where path is a list of PTCAR_IDs and 
                   total_distance is the sum of link distances, or (None, None) if no path exists
        """
        available_links = self.links[self.links['disabled'] == 0]
        shortest_path_layer = folium.FeatureGroup(name='Shortest Path')

        if start_station == end_station:
            return [start_station], 0
       
        n = self.stations.size
        adj_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(adj_matrix, 0)

        # Créer un dictionnaire pour mapper les IDs uniques à leurs indices
        ptcarid_to_index = {id_: index for index, id_ in enumerate(self.stations['PTCAR_ID'])}
        index_to_ptcarid = {index: id_ for index, id_ in enumerate(self.stations['PTCAR_ID'])}

        # Remplir la matrice d'adjacence avec les distances
        for _, row in available_links.iterrows():
            i = ptcarid_to_index[row['Departure_PTCAR_ID']]
            j = ptcarid_to_index[row['Arrival_PTCAR_ID']]
            adj_matrix[i, j] = row["Distance"]

        # Appliquer l'algorithme de Floyd-Warshall avec prédécesseurs
        distances, predecessors = floyd_warshall(adj_matrix, directed=True, return_predecessors=True)
        # Utiliser id_to_index pour obtenir les indices
        start_ptcarid = self.stations.loc[self.stations['Name_FR'] == start_station, 'PTCAR_ID'].iloc[0]
        end_ptcarid = self.stations.loc[self.stations['Name_FR'] == end_station, 'PTCAR_ID'].iloc[0]
        start_idx = ptcarid_to_index[start_ptcarid]
        end_idx = ptcarid_to_index[end_ptcarid]
        path = []
        current = end_idx

        if predecessors[start_idx, end_idx] == -9999:
            return None, None  # Aucun chemin n'existe    
        # Reconstituer le chemin
        while current != start_idx:
            path.append(index_to_ptcarid[current])  # Conserver l'ID d'origine
            current = predecessors[start_idx, current]
        path.append(start_ptcarid)  # Ajouter la station de départ
        # Inverser le chemin pour obtenir de départ à arrivée
        path = path[::-1]

        # Calculer la distance totale du chemin
        total_distance = sum([available_links[(available_links['Departure_PTCAR_ID'] == path[i]) & (available_links['Arrival_PTCAR_ID'] == path[i + 1])]["Distance"].values[0] for i in range(len(path) - 1)])

        return path, total_distance  # Retourner le chemin et la distance totale
    


    def render_shortest_path(self, start_station_name, end_station_name, m, color="#67c9ff"):
        """
        Render the shortest path between two stations on a Folium map.
        
        Args:
            start_station_name (str): Name of the starting station
            end_station_name (str): Name of the destination station
            m (folium.Map): The Folium map object to add the path to
            color (str, optional): Color of the path line. Defaults to "#67c9ff".
            
        Returns:
            tuple: (updated_map, total_distance, path) where updated_map is the map with path rendered,
                   total_distance is the path distance, and path is the list of PTCAR_IDs
        """
        shortest_path_layer = folium.FeatureGroup(name='Shortest Path')

        path, total_distance = self.get_shortest_path(start_station_name, end_station_name)
        if path:
            start_station = self.get_station_by_id(path[0])
            end_station = self.get_station_by_id(path[-1])
            

        start_lat, start_lon = start_station['Geo_Point']
        end_lat, end_lon = end_station['Geo_Point']
        min_lat = min(start_lat, end_lat)
        max_lat = max(start_lat, end_lat) 
        min_lon = min(start_lon, end_lon)
        max_lon = max(start_lon, end_lon)

        lat_margin = (max_lat - min_lat) * 0.05
        lon_margin = (max_lon - min_lon) * 0.05

        m.fit_bounds([[min_lat - lat_margin, min_lon - lon_margin],
                    [max_lat + lat_margin, max_lon + lon_margin]])
            
        self.render_station(start_station,color="#60B2E0",fill_color="white").add_to(shortest_path_layer)
        self.render_station(end_station,color="#60B2E0",fill_color="white").add_to(shortest_path_layer)

        for i in range(len(path) - 1):
            link = self.get_link_by_ids(path[i], path[i + 1])
            self.render_link(link, color=color).add_to(shortest_path_layer)

        shortest_path_layer.add_to(m)
        return m, total_distance, path
         