import pandas as pd
from tempocom.services import DBConnector

class Boxplots:
    def __init__(self, dbc:DBConnector):
        sql_query = """
            SELECT Q1 as q1, Q3 as q3, max, median, min, n_samples, name, outliers, type, planned
            FROM punctuality_boxplots
        """
        self.dbc = dbc
        self.df = pd.DataFrame(dbc.query(sql_query))
        
    def get_unique_relations(self):
        sql_query = """
            SELECT DISTINCT direction_name
            FROM direction_stops
        """
        direction_stops_df = pd.DataFrame(self.dbc.query(sql_query)).drop_duplicates(subset="direction_name").dropna(subset="direction_name")
        return direction_stops_df["direction_name"].str.replace("->", "<->").drop_duplicates().tolist()
    
    def get_boxplots_by_relations(self, relations: list[str],is_two_way: bool = False):
        if is_two_way:
            relation_ids = [relation.split(":")[0] for relation in relations]
            filtered_df = self.df[self.df["name"].str.split(":").str[0].isin(relation_ids)]
            return filtered_df
        else:
            return self.df[self.df["name"].isin(relations)]

    def get_stations_from_relation(self, relation: str):
        relation = relation.replace("<->", "->")
        sql_query =f"""
            SELECT DISTINCT station_name, order_in_route
            FROM direction_stops
            WHERE direction_name = '{relation}'
            ORDER BY order_in_route
        """
        stations_df = pd.DataFrame(self.dbc.query(sql_query)).drop_duplicates(subset="station_name").dropna(subset="station_name")
        return stations_df["station_name"].tolist()

    def get_boxplots_by_stations(self, stations: list[str]):
        stations_str = "', '".join(stations)
        sql_query =f"""
            SELECT Q1 as q1, Q3 as q3, max, median, min, n_samples, name, outliers, type, planned
            FROM punctuality_boxplots
            WHERE name IN ('{stations_str}')
        """
        stations_boxplots_df = pd.DataFrame(self.dbc.query(sql_query))
        return stations_boxplots_df
    