import pandas as pd
from utils import get_mart
import folium
from objects.MacroNetwork import MacroNetwork
import ast
import streamlit as st
import logging
import time

class Coupures:

    PALETTES = {
            "CTL": {"color": "#FF0000", "dash": None, "label": "CTL"},
            "Keep Free": {"color": "#0066FF", "dash": None, "label": "Keep Free"},
            "SAVU": {"color": "#FF0000", "dash": "5,8", "label": "SAVU"},
            "Travaux possibles": {"color": "#FFA500", "dash": None, "label": "Travaux possibles"},
            "OTHER": {"color": "#fd6c9e", "dash": None, "label": "Autre impact"},
        }

    def __init__(self,path_to_mart:str='./mart'):
        self.coupures = get_mart(f'{path_to_mart}/private/colt.csv')
        self.opdf = get_mart(f'{path_to_mart}/public/opdf.csv')
        self.descriptions = get_mart(f'{path_to_mart}/private/colt_descriptions.csv')

        self.coupures['section_from_id'] = pd.to_numeric(self.coupures['section_from_id'], errors='coerce').fillna(-1).astype(int)
        self.coupures['section_to_id'] = pd.to_numeric(self.coupures['section_to_id'], errors='coerce').fillna(-1).astype(int)
        self.coupures['date_begin'] = pd.to_datetime(self.coupures['date_begin'], format='%Y-%m-%d')
        self.coupures['date_end'] = pd.to_datetime(self.coupures['date_end'], format='%Y-%m-%d')
        self.coupures['time_begin'] = pd.to_datetime(self.coupures['time_begin'], format='%H:%M:%S', errors='coerce').dt.time
        self.coupures['time_end'] = pd.to_datetime(self.coupures['time_end'], format='%H:%M:%S', errors='coerce').dt.time
        self.opdf['Geo_Point'] = self.opdf['Geo_Point'].apply(lambda x: ast.literal_eval(x))

        #filter form
        self.status = self.coupures['status'].dropna().sort_values().unique()
        self.period_type = self.coupures['period_type'].dropna().sort_values().unique()
        self.impact = self.coupures['impact'].dropna().sort_values().unique()
        self.leaders = self.coupures['leader'].dropna().sort_values().unique()

        
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
        lat, lon = op['Geo_Point']
        text = f"{op['Abbreviation_BTS_French_complete']}(ID: {op['PTCAR_ID']}) - {op['Classification_FR']}"
        return folium.CircleMarker(
            [lat, lon],
            color="orange",
            fill=True,
            fill_color="yellow",
            radius=2,
            popup=text,
            tooltip=text
        )

    def render_coupure(self, cou_id, network: MacroNetwork, opacity=1, line_weight=7, layer_name='Coupures'):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        start_time = time.time()
        logger.debug("▶️ Début render_coupure - cou_id=%s, layer_name=%s", cou_id, layer_name)

        CoupureLayer = folium.FeatureGroup(name=layer_name)
        coupure = self.coupures[self.coupures['cou_id'] == cou_id]

        logger.debug("Nombre de lignes de coupure filtrées : %d", len(coupure))

        impact_map = {
            "CTL": "CTL",
            "Keep Free": "Keep Free",
            "SAVU": "SAVU"
        }

        added_lines = 0
        added_markers = 0

        for idx, row in coupure.iterrows():
            logger.debug("Traitement ligne index=%d", idx)

            if pd.isna(row['section_from_id']) or pd.isna(row['section_to_id']):
                logger.debug("⛔ section_from_id ou section_to_id manquant → skip")
                continue

            impact_key = impact_map.get(row['impact'], "OTHER")
            style = self.PALETTES.get(impact_key, {"color": "gray", "dash": None})
            line_kw = dict(color=style["color"], weight=line_weight, opacity=opacity, dash_array=style["dash"])
            logger.debug("Impact: %s → style: %s", row['impact'], style)

            if self.both_sections_exists_on_macro_network(row, network):
                section_from = network.get_station_by_id(row['section_from_id'])
                section_to = network.get_station_by_id(row['section_to_id'])

                if section_from is None or section_to is None:
                    logger.debug("❗ Station introuvable dans le réseau macro")
                    continue

                logger.debug("↔️ Recherche plus court chemin : %s → %s", section_from['Name_FR'], section_to['Name_FR'])

                path, _ = network.get_shortest_path(section_from['Name_FR'], section_to['Name_FR'])

                if path is not None:
                    for i in range(len(path) - 1):
                        link = network.get_link_by_ids(path[i], path[i + 1])
                        if link is not None and 'Geo_Shape' in link:
                            folium.PolyLine(link['Geo_Shape'], **line_kw).add_to(CoupureLayer)
                            added_lines += 1
                        else:
                            logger.debug("❓ Lien absent pour %s → %s", path[i], path[i+1])
            else:
                logger.debug("⚠️ Sections absentes du réseau macro, fallback sur Geo_Point")
                op_from = self.get_opdf_by_id(row['section_from_id'])
                op_to = self.get_opdf_by_id(row['section_to_id'])

                if op_from is None or op_to is None:
                    logger.debug("❌ op_from ou op_to introuvable")
                    continue

                try:
                    lat1, lon1 = op_from['Geo_Point']
                    lat2, lon2 = op_to['Geo_Point']
                    folium.PolyLine([[lat1, lon1], [lat2, lon2]], **line_kw).add_to(CoupureLayer)
                    added_lines += 1

                    for op_id in [row['section_from_id'], row['section_to_id']]:
                        op_marker = self.render_op(op_id)
                        if op_marker:
                            op_marker.add_to(CoupureLayer)
                            added_markers += 1

                    folium.Marker(
                        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
                        icon=folium.DivIcon(html="<span style='color:yellow;font-size:18px;'>[!]</span>"),
                        tooltip="Lien absent du réseau réel"
                    ).add_to(CoupureLayer)
                    added_markers += 1

                except Exception as e:
                    logger.exception("💥 Erreur lors du parsing Geo_Point : %s", e)
                    continue

            elapsed = time.time() - start_time
            logger.debug("✅ Fin render_coupure — lignes ajoutées: %d, marqueurs: %d, durée: %.2fs", added_lines, added_markers, elapsed)
        return CoupureLayer
        
    def render_contextual_coupures(self, cou_id, network: MacroNetwork):
        coupure = self.coupures[self.coupures['cou_id'] == cou_id]
        coupure = coupure[coupure['status'] == 'Y'].drop_duplicates('cou_id')
        
        if coupure.empty:
            return None
            
        current_date_begin = pd.to_datetime(coupure.iloc[0]['date_begin'])
        current_date_end = pd.to_datetime(coupure.iloc[0]['date_end'])
        
        date_overlapping = self.coupures[
            ((self.coupures['date_begin'] <= current_date_end) & 
            (self.coupures['date_end'] >= current_date_begin)) &
            (self.coupures['cou_id'] != cou_id)
        ].drop_duplicates('cou_id')
        
        current_time_begin = pd.to_datetime('00:00:00').time()
        current_time_end = pd.to_datetime('00:00:00').time()
        
        if pd.notna(coupure.iloc[0]['time_begin']):
            current_time_begin = pd.to_datetime(str(coupure.iloc[0]['time_begin'])).time()
        if pd.notna(coupure.iloc[0]['time_end']):
            current_time_end = pd.to_datetime(str(coupure.iloc[0]['time_end'])).time()
            
        time_overlapping = date_overlapping[
            ((date_overlapping['time_begin'].fillna('00:00:00').apply(lambda x: pd.to_datetime(str(x)).time()) <= current_time_end) &
            (date_overlapping['time_end'].fillna('00:00:00').apply(lambda x: pd.to_datetime(str(x)).time()) >= current_time_begin))
        ]
        
        if time_overlapping.empty:
            return None
            
        CoupureLayer = None
        for _, row in time_overlapping.iterrows():
            layer = self.render_coupure(row['cou_id'], network, opacity=1, line_weight=3, layer_name='Contextual Coupures')
            if layer:
                if CoupureLayer is None:
                    CoupureLayer = layer
                else:
                    for child in layer._children.values():
                        child.add_to(CoupureLayer)

        return CoupureLayer
            
    def get_cou_id_list_by_filter(self, filter):
        df = self.coupures.copy()
        
        for key, value in filter.items():
            if value:
                if key == 'cou_id': 
                    df = df[df[key].astype(str).str.contains(value, case=False, na=False)]
                else: 
                    df = df[df[key].isin(value)]
        
        return df['cou_id'].unique().tolist()
    
    def both_sections_exists_on_macro_network(self, cou_id, network: MacroNetwork):
        return (cou_id['section_from_id'] in network.stations['PTCAR_ID'].values and 
                cou_id['section_to_id'] in network.stations['PTCAR_ID'].values)

    
    def get_unique_coupure_types(self,selected_columns):
        unique_coupure_types = self.coupures.groupby(selected_columns).size().reset_index(name='count')
        return unique_coupure_types