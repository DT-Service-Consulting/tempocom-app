from features.macro_network.MacroLinks import MacroLinks
from features.operational_points.OperationalPoints import OperationalPoints
import folium
from tempocom.services.DBConnector import DBConnector
import pandas as pd

class MacroNetworkView:

    def get_macro_links_layer(self, macro_links:MacroLinks, oper_points:OperationalPoints) -> folium.FeatureGroup:
        layer = folium.FeatureGroup(name="Macro Links")
        processed_links = set()
        
        for _, row in macro_links.df.iterrows():
            link_key = tuple(sorted([row['DEPARTURE_ID'], row['ARRIVAL_ID']]))
            
            if link_key in processed_links:
                continue
            
            processed_links.add(link_key)
            
            color = 'red' if row.get('DISABLED', False) else 'grey'
            display_text = f"{oper_points.get_name_by_id(row['DEPARTURE_ID'])} ⇔ {oper_points.get_name_by_id(row['ARRIVAL_ID'])} ({int(row['DISTANCE'])} km)"
            
            folium.PolyLine(
                locations=row['GEO_SHAPE'],
                color=color,
                weight=3,
                popup=display_text,
                tooltip=display_text
            ).add_to(layer)
        return layer

    def get_disabled_macro_links_layer(self, disabled_links_df: pd.DataFrame, oper_points: OperationalPoints) -> folium.FeatureGroup:
        layer = folium.FeatureGroup(name="Disabled Links")
        processed_links = set()
        
        for _, row in disabled_links_df.iterrows():
            link_key = tuple(sorted([row['DEPARTURE_ID'], row['ARRIVAL_ID']]))
            
            if link_key in processed_links:
                continue
            
            processed_links.add(link_key)
            
            display_text = f"CLOSED: {oper_points.get_name_by_id(row['DEPARTURE_ID'])} ⇔ {oper_points.get_name_by_id(row['ARRIVAL_ID'])} ({int(row['DISTANCE'])} km)"
            
            folium.PolyLine(
                locations=row['GEO_SHAPE'],
                color='red',
                weight=5,
                dash_array='10,5',
                popup=display_text,
                tooltip=display_text
            ).add_to(layer)
        return layer

    