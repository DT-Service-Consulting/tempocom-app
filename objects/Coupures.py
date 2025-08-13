from utils import get_mart
from objects.MacroNetwork import MacroNetwork
import logging,time,os,ast,folium, pandas as pd, streamlit as st, utils, sys

class Coupures:
    """
    A class for managing and visualizing railway disruptions (coupures) data.
    
    This class handles railway disruption data including CTL (Contrôle Technique de Ligne),
    Keep Free, SAVU, and other impact types. It provides methods for filtering, rendering,
    and analyzing disruption data on railway networks.
    
    Attributes:
        coupures (DataFrame): Main disruption data
        opdf (DataFrame): Operational points data
        descriptions (DataFrame): Disruption descriptions
        dat (DataFrame): DAT (Données d'Analyse Technique) model data
        ctl_sections (DataFrame): CTL sections data
        status (array): Unique status values for filtering
        period_type (array): Unique period types for filtering
        impact (array): Unique impact types for filtering
        leaders (array): Unique leader values for filtering
    """

    PALETTES = {
            "CTL": {"color": "#FF0000", "dash": None, "label": "CTL"},
            "Keep Free": {"color": "#0066FF", "dash": None, "label": "Keep Free"},
            "SAVU": {"color": "#FF0000", "dash": "5,8", "label": "SAVU"},
            "Travaux possibles": {"color": "#FFA500", "dash": None, "label": "Travaux possibles"},
            "OTHER": {"color": "#fd6c9e", "dash": None, "label": "Autre impact"},
        }

    def __init__(self,path_to_mart:str="../tempocom_mart"):
        """
        Initialize the Coupures object with data from the mart directory.
        
        Args:
            path_to_mart (str): Path to the mart directory containing data files.
                Defaults to MART_RELATIVE_PATH environment variable.
                
        Note:
            Loads and preprocesses various CSV files including coupures, opdf,
            descriptions, dat model, and ctl sections. Converts data types and
            extracts unique filter values.
        """
        self.coupures = get_mart(f'{path_to_mart}/private/colt.csv')
        self.opdf = get_mart(f'{path_to_mart}/public/opdf.csv')
        self.descriptions = get_mart(f'{path_to_mart}/private/colt_descriptions.csv')
        self.dat = get_mart(f'{path_to_mart}/private/colt_dat_S1_model.csv')
        self.ctl_sections = pd.read_csv(f'{path_to_mart}/private/ctl_sections.csv')

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


    def get_ctl_sections(self):
        """
        Get unique CTL section combinations from the DAT model.
        
        Returns:
            list: List of CTL section combinations in format "From ⇔ To"
        """
        ctl_combinations = []
        for _, row in self.dat.iterrows():
            ctl_from_abbrev = self.opdf[self.opdf['PTCAR_ID'] == row['ctl_from']]['Abbreviation_BTS_French_complete'].iloc[0] if not self.opdf[self.opdf['PTCAR_ID'] == row['ctl_from']].empty else f"ID_{row['ctl_from']}"
            ctl_to_abbrev = self.opdf[self.opdf['PTCAR_ID'] == row['ctl_to']]['Abbreviation_BTS_French_complete'].iloc[0] if not self.opdf[self.opdf['PTCAR_ID'] == row['ctl_to']].empty else f"ID_{row['ctl_to']}"
            combination = f"{ctl_from_abbrev} ⇔ {ctl_to_abbrev}"
            if combination not in ctl_combinations:
                ctl_combinations.append(combination)
        return ctl_combinations

    def categorize_impact(self, impact):
        """
        Categorize impact types into predefined categories.
        
        Args:
            impact: The impact value to categorize
            
        Returns:
            str: Categorized impact type ('CTL', 'Keep Free', 'SAVU', 'Travaux possibles', or 'Autre')
        """
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
        """
        Get operational point data by PTCAR_ID.
        
        Args:
            id: The PTCAR_ID to search for
            
        Returns:
            Series or None: Operational point data if found, None otherwise
        """
        result = self.opdf[self.opdf['PTCAR_ID'] == id]
        if not result.empty:
            return result.iloc[0]
        else:
            return None
    
    def render_op(self, opdf_id):
        """
        Render an operational point as a Folium marker.
        
        Args:
            opdf_id: The PTCAR_ID of the operational point
            
        Returns:
            folium.CircleMarker or None: Folium marker for the operational point, None if not found
        """
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
        """
        Render a specific disruption on the network map.
        
        Args:
            cou_id: The disruption ID to render
            network (MacroNetwork): The network object containing station and link data
            opacity (float, optional): Line opacity. Defaults to 1.
            line_weight (int, optional): Line weight. Defaults to 7.
            layer_name (str, optional): Name of the Folium layer. Defaults to 'Coupures'.
            
        Returns:
            folium.FeatureGroup: Folium layer containing the rendered disruption
        """
        coupure = self.coupures[self.coupures['cou_id'] == cou_id]
        return self.render_coupure_line(coupure, network, opacity, line_weight, layer_name)

    def render_coupure_line(self, coupure, network: MacroNetwork, opacity=1, line_weight=7, layer_name='Coupures'):
        """
        Render disruption lines on the network map.
        
        Args:
            coupure (DataFrame): Disruption data to render
            network (MacroNetwork): The network object containing station and link data
            opacity (float, optional): Line opacity. Defaults to 1.
            line_weight (int, optional): Line weight. Defaults to 7.
            layer_name (str, optional): Name of the Folium layer. Defaults to 'Coupures'.
            
        Returns:
            folium.FeatureGroup: Folium layer containing the rendered disruption lines
        """
        CoupureLayer = folium.FeatureGroup(name=layer_name)
        impact_map = {
            "CTL": "CTL",
            "Keep Free": "Keep Free",
            "SAVU": "SAVU"
        }

        added_lines = 0
        added_markers = 0

        for idx, row in coupure.iterrows():
            if pd.isna(row['section_from_id']) or pd.isna(row['section_to_id']):

                continue

            impact_key = impact_map.get(row['impact'], "OTHER")
            style = self.PALETTES.get(impact_key, {"color": "gray", "dash": None})
            line_kw = dict(color=style["color"], weight=line_weight, opacity=opacity, dash_array=style["dash"])


            if self.both_sections_exists_on_macro_network(row, network):
                section_from = network.get_station_by_id(row['section_from_id'])
                section_to = network.get_station_by_id(row['section_to_id'])

                if section_from is None or section_to is None:

                    continue


                path, _ = network.get_shortest_path(section_from['Name_FR'], section_to['Name_FR'])

                if path is not None:
                    for i in range(len(path) - 1):
                        link = network.get_link_by_ids(path[i], path[i + 1])
                        if link is not None and 'Geo_Shape' in link:
                            folium.PolyLine(link['Geo_Shape'], **line_kw).add_to(CoupureLayer)
                            added_lines += 1
                        else:
                            continue
            else:
                op_from = self.get_opdf_by_id(row['section_from_id'])
                op_to = self.get_opdf_by_id(row['section_to_id'])

                if op_from is None or op_to is None:
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
                        tooltip="Link absent from macro network"
                    ).add_to(CoupureLayer)
                    added_markers += 1

                except Exception as e:
                    continue
        return CoupureLayer
        
    def render_contextual_coupures(self, cou_id, network: MacroNetwork):
        """
        Render disruptions that overlap in time with a specific disruption.
        
        Args:
            cou_id: The reference disruption ID
            network (MacroNetwork): The network object containing station and link data
            
        Returns:
            folium.FeatureGroup or None: Folium layer containing overlapping disruptions, None if none found
        """
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
        """
        Get list of disruption IDs based on filter criteria.
        
        Args:
            filter (dict): Dictionary containing filter criteria with column names as keys
            
        Returns:
            list: List of disruption IDs matching the filter criteria
        """
        df = self.coupures.copy()
        
        for key, value in filter.items():
            if value:
                if key == 'cou_id': 
                    df = df[df[key].astype(str).str.contains(value, case=False, na=False)]
                else: 
                    df = df[df[key].isin(value)]
        
        return df['cou_id'].unique().tolist()
    
    def both_sections_exists_on_macro_network(self, cou_id, network: MacroNetwork):
        """
        Check if both sections of a disruption exist in the macro network.
        
        Args:
            cou_id: Disruption data row containing section_from_id and section_to_id
            network (MacroNetwork): The network object to check against
            
        Returns:
            bool: True if both sections exist in the network, False otherwise
        """
        return (cou_id['section_from_id'] in network.stations['PTCAR_ID'].values and 
                cou_id['section_to_id'] in network.stations['PTCAR_ID'].values)

    
    def get_unique_coupure_types(self,selected_columns):
        """
        Get unique disruption types grouped by selected columns.
        
        Args:
            selected_columns (list): List of column names to group by
            
        Returns:
            DataFrame: DataFrame containing unique combinations and their counts
        """
        unique_coupure_types = self.coupures.groupby(selected_columns).size().reset_index(name='count')
        return unique_coupure_types
    
    def get_kf_pred(self, nb_occ):
        """
        Get Keep Free prediction emoji based on occurrence count.
        
        Args:
            nb_occ (int): Number of occurrences
            
        Returns:
            str: Emoji representing the prediction level (💎, 🥇, 🥈, 🥉)
        """
        if nb_occ >= 10:
            return "💎"
        elif nb_occ >= 5:
            return "🥇"
        elif nb_occ >= 3:
            return "🥈"
        else:
            return "🥉"
    
    def advise_keepfrees(self, ctl_section, network: MacroNetwork):
        """
        Generate Keep Free advice for a specific CTL section.
        
        Args:
            ctl_section (str): CTL section in format "From <=> To"
            network (MacroNetwork): The network object (not used in current implementation)
            
        Returns:
            list: List of dictionaries containing advised disruptions with their details
        """
        section_from_name, section_to_name = ctl_section.split(" <=> ")
        section_from_id = self.opdf[self.opdf['Abbreviation_BTS_French_complete'] == section_from_name]['PTCAR_ID'].iloc[0]
        section_to_id = self.opdf[self.opdf['Abbreviation_BTS_French_complete'] == section_to_name]['PTCAR_ID'].iloc[0]
        df = self.dat
        mask1 = (df['ctl_from'] == section_from_id) & (df['ctl_to'] == section_to_id)
        mask2 = (df['ctl_from'] == section_to_id) & (df['ctl_to'] == section_from_id)
        keep_free = df[mask1 | mask2]
        
        advised_coupures = []
        advised_ctl = {
            'cou_id': f"advised_ctl_{section_from_id}_{section_to_id}",
            'section_from_id': section_from_id,
            'section_to_id': section_to_id,
            'section_from_name': section_from_name,
            'section_to_name': section_to_name,
            'impact': 'CTL',
            'nb_occ': len(keep_free)
        }
        advised_coupures.append(advised_ctl)
        for _, row in keep_free.iterrows():
            if row['nb_occ'] < 2:
                continue
            advised_coupure = {
                'cou_id': f"advised_{row['kf_from']}_{row['kf_to']}",
                'section_from_id': row['kf_from'],
                'section_to_id': row['kf_to'],
                'impact': 'Keep Free',
                'nb_occ': self.get_kf_pred(row['nb_occ']),
                'section_from_name': self.opdf[self.opdf['PTCAR_ID'] == row['kf_from']]['Abbreviation_BTS_French_complete'].iloc[0],
                'section_to_name': self.opdf[self.opdf['PTCAR_ID'] == row['kf_to']]['Abbreviation_BTS_French_complete'].iloc[0]
            }
            advised_coupures.append(advised_coupure)
        
        return advised_coupures