from tempocom.services import DBConnector
import polars as pl
import pydeck as pdk
import pandas as pd
import ast
import folium

class OperationalPoints:
    sql_query="""
        SELECT
        PTCAR_ID as ID,
        Geo_Point as GEO_POINT,
        Complete_name_in_French as NAME,
        Complete_name_in_Dutch as NAME_NL,
        Classification_EN as CLASS,
        CAST(0 AS BIT) as ISIN_MACRO_NETWORK,
        NULL as N_LINKS
        FROM operational_points
        """

    def load(self, lang = "EN"):
        dbc = DBConnector()
        self.df = pd.read_sql(self.sql_query, dbc.connect())
        self.df['NAME'] = self.df.apply(
            lambda row: row['NAME'] if row['NAME'] == row['NAME_NL'] else f"{row['NAME']} / {row['NAME_NL']}", 
            axis=1
        )
        self.df['GEO_POINT'] = self.df['GEO_POINT'].apply(lambda x: tuple(map(float, ast.literal_eval(x))))
        return self

    def get_point_by_id(self,PTCAR_ID):
        point = self.df[self.df['ID'] == PTCAR_ID]
        if point.empty:
            return None
        return point.iloc[0].to_dict()
    
    def get_type_by_id(self,PTCAR_ID):
        point = self.get_point_by_id(PTCAR_ID)
        return point['CLASS']
    
    def get_geopoint_by_id(self,PTCAR_ID):
        point = self.get_point_by_id(PTCAR_ID)
        return point['GEO_POINT']
    
    def get_name_by_id(self,PTCAR_ID):
        point = self.get_point_by_id(PTCAR_ID)
        return point['NAME']
    
    def get_id_by_name(self, name):
        point = self.df[self.df['NAME'] == name]
        if point.empty:
            return None
        return point.iloc[0]['ID']
    
    def get_station_names(self):
        return self.df['NAME'].tolist()



