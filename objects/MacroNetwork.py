from utils import get_mart
import folium
import pandas as pd
import ast
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

class MacroNetwork:
    def __init__(self,path_to_mart:str='./mart'):      
        # extracting
        self.links = get_mart(f'{path_to_mart}/public/network_graph.csv')
        self.stations = get_mart(f'{path_to_mart}/public/stations.csv')
        #self.available_links = self.links[self.links['disabled'] == 0]
        #processing
        self.compute_number_of_links()
        self.links['Distance'] = self.links['Distance'].apply(lambda x: round(float(x), 1))

    def get_station_by_id(self, ptcarid):
        return self.stations[self.stations['PTCAR_ID'] == ptcarid].iloc[0]
    
    def get_link_by_ids(self, id1, id2):
        return self.links[(self.links['Departure_PTCAR_ID'] == id1) & (self.links['Arrival_PTCAR_ID'] == id2)
                          | (self.links['Departure_PTCAR_ID'] == id2) & (self.links['Arrival_PTCAR_ID'] == id1)].iloc[0]
    
    def get_open_links(self):
        open_links = self.links[self.links['disabled'] == 0]
        connections = []
        for _, link in open_links.iterrows():
            conn = f"{link['Departure_Name_FR']} ⇔ {link['Arrival_Name_FR']}"
            reverse_conn = f"{link['Arrival_Name_FR']} ⇔ {link['Departure_Name_FR']}"
            if conn not in connections and reverse_conn not in connections:
                connections.append(conn)
        return connections
    
    def render_link(self,link,color="grey"):
        text = f"{link['Departure_Name_FR']} ⇔ {link['Arrival_Name_FR']} ({link['Distance']} km)"
        return folium.PolyLine(
            locations=[ast.literal_eval(link['Geo_Shape'])],
            color=color,
            weight=3,
            popup=text,
            tooltip=text
        )
    
    def render_station(self,station,color="white",fill_color="black"):
        text = f"{station['Name_FR']} ({station['n_links']} links) - {station['Classification_FR']}"
        # Rayon : 1 si <=2 connexions, sinon 3
        radius = 1 if station['n_links'] <= 2 else 3
        return folium.CircleMarker(
            location=ast.literal_eval(station['Geo_Point']),
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            tooltip=text,
            popup=text
        )
    
    def compute_number_of_links(self):
        self.stations['n_links'] = 0
        
        for _, link in self.links.iterrows():
            dep_id = link['Departure_PTCAR_ID']
            arr_id = link['Arrival_PTCAR_ID']
            
            self.stations.loc[self.stations['PTCAR_ID'] == dep_id, 'n_links'] += 1
            self.stations.loc[self.stations['PTCAR_ID'] == arr_id, 'n_links'] += 1
            
        self.stations['n_links'] = self.stations['n_links'] / 2
        self.stations['n_links'] = self.stations['n_links'].astype(int)

    def render_macro_network(self,m):
        links_layer = folium.FeatureGroup(name='Links')
        stations_layer = folium.FeatureGroup(name='Stations')

        #render links
        for _, link in self.links.iterrows():
            self.render_link(link).add_to(links_layer)

        #render stations
        for _, info in self.stations.iterrows():
            self.render_station(info).add_to(stations_layer)

        links_layer.add_to(m)
        stations_layer.add_to(m)

        return m
    
    def close_links(self, closed_links: list[tuple], m):
        edited_network = folium.FeatureGroup(name='Closed Links')

        #disable both ways
        for link in closed_links:
            mask_way1 = (self.links['Departure_Name_FR'] == link[0]) & (self.links['Arrival_Name_FR'] == link[1])
            self.links.loc[mask_way1, 'disabled'] = 1
            mask_way2 = (self.links['Departure_Name_FR'] == link[1]) & (self.links['Arrival_Name_FR'] == link[0])
            self.links.loc[mask_way2, 'disabled'] = 1

        #render edited network  
        for _, link in self.links.loc[self.links['disabled'] == 1].iterrows():
            self.render_link(link,color="red").add_to(edited_network)
        edited_network.add_to(m)
        self.available_links = self.links[self.links['disabled'] == 0]
        return m
            

    def get_shortest_path(self, start_station, end_station):
        available_links = self.links[self.links['disabled'] == 0]
        shortest_path_layer = folium.FeatureGroup(name='Shortest Path')
       
        n = self.stations.size
        adj_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(adj_matrix, 0)

        # Créer un dictionnaire pour mapper les IDs uniques à leurs indices
        ptcarid_to_index = {id_: index for index, id_ in enumerate(self.stations['PTCAR_ID'])}
        index_to_ptcarid = {index: id_ for index, id_ in enumerate(self.stations['PTCAR_ID'])}

        # Remplir la matrice d'adjacence avec les distances
        for _, row in available_links.iterrows():
            i = ptcarid_to_index[int(row['Departure_PTCAR_ID'])]
            j = ptcarid_to_index[int(row['Arrival_PTCAR_ID'])]
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
        shortest_path_layer = folium.FeatureGroup(name='Shortest Path')

        path, total_distance = self.get_shortest_path(start_station_name, end_station_name)
        if path:
            start_station = self.get_station_by_id(path[0])
            end_station = self.get_station_by_id(path[-1])
            

        start_lat, start_lon = ast.literal_eval(start_station['Geo_Point'])
        end_lat, end_lon = ast.literal_eval(end_station['Geo_Point'])
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
         