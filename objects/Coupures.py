import pandas as pd
from utils import get_mart
import folium
from objects.MacroNetwork import MacroNetwork
import ast
import streamlit as st

class Coupures:

    PALETTES = {
            "CTL": {"color": "#FF0000", "dash": None, "label": "CTL"},
            "Keep Free": {"color": "#0066FF", "dash": None, "label": "Keep Free"},
            "SAVU": {"color": "#FF0000", "dash": "5,8", "label": "SAVU"},
            "Travaux possibles": {"color": "#FFA500", "dash": None, "label": "Travaux possibles"},
            "OTHER": {"color": "#800080", "dash": None, "label": "Autre impact"},
        }

    def __init__(self):
        self.coupures = get_mart('./mart/private/colt.csv')
        self.opdf = get_mart('./mart/public/opdf.csv')
        self.descriptions = get_mart('./mart/private/colt_descriptions.csv')

        self.leaders = self.coupures['leader'].unique()
        self.coupures['section_from_id'] = pd.to_numeric(self.coupures['section_from_id'], errors='coerce').fillna(-1).astype(int)
        self.coupures['section_to_id'] = pd.to_numeric(self.coupures['section_to_id'], errors='coerce').fillna(-1).astype(int)
        self.coupures['date_begin'] = pd.to_datetime(self.coupures['date_begin'], format='%Y-%m-%d')
        self.coupures['date_end'] = pd.to_datetime(self.coupures['date_end'], format='%Y-%m-%d')
        self.coupures['time_begin'] = pd.to_datetime(self.coupures['time_begin'], format='%H:%M:%S', errors='coerce').dt.time
        self.coupures['time_end'] = pd.to_datetime(self.coupures['time_end'], format='%H:%M:%S', errors='coerce').dt.time
        
        self.coupures['impact'] = self.coupures['impact'].apply(self.categorize_impact)

    def categorize_impact(self, impact):
        if pd.isna(impact):
            return 'Autre'
        impact = str(impact)
        if 'CTL' in impact:
            return 'CTL'
        elif 'Keep Free' in impact:
            return 'Keep Free'
        elif 'SAVU' in impact:
            return 'SAVU'
        elif 'Travaux possibles' in impact:
            return 'Travaux possibles'
        else:
            return 'Autre'

    def get_opdf_by_id(self, id):
        result = self.opdf[self.opdf['PTCAR_ID'] == id]
        if not result.empty:
            return result.iloc[0]
        else:
            return None
    
    def render_op(self, opdf_id):
        op = self.get_opdf_by_id(opdf_id)
        if op is None:
            return None
        lat, lon = map(float, op['Geo_Point'].split(","))
        text = f"{op['Abbreviation_BTS_French_complete']} - {op['Classification_FR']}"
        return folium.CircleMarker(
            [lat, lon],
            color="orange",
            fill=True,
            fill_color="yellow",
            radius=2,
            popup=text,
            tooltip=text
        )

    def render_coupure(self, m: folium.Map, cou_id, network: MacroNetwork):
        CoupureLayer = folium.FeatureGroup(name='Coupures')
        coupure = self.coupures[self.coupures['cou_id'] == cou_id]
        
        for _, row in coupure.iterrows():
            if pd.isna(row['section_from_id']) or pd.isna(row['section_to_id']):
                st.write("One of the sections cannot be rendered")
                continue

            impact_key = "CTL" if "CTL" in row['impact'] else "Keep Free" if "Keep Free" in row['impact'] else "SAVU" if "SAVU" in row['impact'] else "OTHER"

            style = self.PALETTES[impact_key]
            line_kw = dict(color=style["color"], weight=4, opacity=0.95, dash_array=style["dash"])
            
            if self.both_sections_exists_on_macro_network(row, network):
                section_from_Name_FR = network.get_station_by_id(row['section_from_id'])['Name_FR']
                section_to_Name_FR = network.get_station_by_id(row['section_to_id'])['Name_FR']
                path, _ = network.get_shortest_path(section_from_Name_FR, section_to_Name_FR)
                
                for i in range(len(path) - 1):
                    link = network.get_link_by_ids(path[i], path[i + 1])
                    folium.PolyLine(ast.literal_eval(link['Geo_Shape']), **line_kw).add_to(CoupureLayer)
            else:
                op_from = self.get_opdf_by_id(row['section_from_id'])
                op_to = self.get_opdf_by_id(row['section_to_id'])
                if op_from is None or op_to is None:
                    continue
                lat1, lon1 = map(float, op_from['Geo_Point'].split(","))
                lat2, lon2 = map(float, op_to['Geo_Point'].split(","))
                
                folium.PolyLine([[lat1, lon1], [lat2, lon2]], **line_kw).add_to(CoupureLayer)
                op_marker_from = self.render_op(row['section_from_id'])
                op_marker_to = self.render_op(row['section_to_id'])
                if op_marker_from:
                    op_marker_from.add_to(CoupureLayer)
                if op_marker_to:
                    op_marker_to.add_to(CoupureLayer)
                
                folium.Marker(
                    location=[(lat1+lat2)/2, (lon1+lon2)/2],
                    icon=folium.DivIcon(html="<span style='color:yellow;font-size:18px;'>⚠</span>"),
                    tooltip="Lien absent du réseau réel"
                ).add_to(CoupureLayer)
                
        CoupureLayer.add_to(m)
        return m
    
    def both_sections_exists_on_macro_network(self, cou_id, network: MacroNetwork):
        return (cou_id['section_from_id'] in network.stations['PTCAR_ID'].values and 
                cou_id['section_to_id'] in network.stations['PTCAR_ID'].values)