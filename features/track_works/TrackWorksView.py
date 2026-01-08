import folium
import pandas as pd
from features.track_works.TrackWorks import TrackWorks
from features.operational_points.OperationalPoints import OperationalPoints
from features.macro_network.MacroLinks import MacroLinks
from datetime import datetime

class TrackWorksView:
    impact_colors = {
        "CTL": {"color": "#FF0000", "dash": None, "label": "CTL"},
        "Keep Free": {"color": "#0066FF", "dash": None, "label": "Keep Free"},
        "SAVU": {"color": "#FF0000", "dash": "5,8", "label": "SAVU"},
        "Travaux possibles": {"color": "#FFA500", "dash": None, "label": "Travaux possibles"},
        "OTHER": {"color": "#fd6c9e", "dash": None, "label": "Autre impact"},
    }
    
    def get_track_works_layer(self, track_works: TrackWorks, oper_points: OperationalPoints, macro_links: MacroLinks, date: datetime):
        layer = folium.FeatureGroup(name="Track Works")
        
        if track_works.df is None or len(track_works.df) == 0:
            return layer
            
        for _, row in track_works.df.iterrows():
            if not track_works.has_valid_sections(row):
                continue
                
            impact_config = self.impact_colors.get(str(row['impact']), self.impact_colors["OTHER"])
            impact_desc = f"({row['impact']}) Track works ID: {row['cou_id']}"
            if track_works.is_point_work(row):
                self._add_point_marker(layer, row, oper_points, impact_config, impact_desc)
            else:
                self._add_line_work(layer, row, oper_points, macro_links, impact_config, impact_desc)
                
        return layer
    
    def _add_point_marker(self, layer, row, oper_points:OperationalPoints, impact_config, impact_desc):
        from_point = oper_points.get_geopoint_by_id(int(float(row['section_from_id'])))
        folium.CircleMarker(
            location=from_point, 
            color=impact_config['color'], 
            weight=3, 
            radius=5,
            popup=impact_desc,
            tooltip=impact_desc
        ).add_to(layer)
    
    def _add_line_work(self, layer, row, oper_points:OperationalPoints, macro_links:MacroLinks, impact_config, impact_desc):
        from_point = oper_points.get_geopoint_by_id(row['section_from_id'])
        to_point = oper_points.get_geopoint_by_id(row['section_to_id'])
        
        if macro_links.isin_macro_network(row['section_from_id']) and macro_links.isin_macro_network(row['section_to_id']):
            locations = macro_links.get_shortest_path_coordinates(row['section_from_id'], row['section_to_id'])
        else:
            locations = [from_point, to_point]

        folium.PolyLine(
            locations=locations,
            color=impact_config['color'],
            weight=3,
            dash_array=impact_config['dash'],
            popup=impact_desc,
            tooltip=impact_desc
        ).add_to(layer)



    def get_table_view(self, track_works: TrackWorks, oper_points: OperationalPoints):
        import streamlit as st
        st.title("Active Track Works")
        if track_works.df is None or len(track_works.df) == 0:
            st.info("🚧 No active track works found")
            return
        
        # Group works by cou_id
        ids = sorted(track_works.df['cou_id'].unique())
        descriptions = track_works.get_descriptions(ids)
        for id in ids:
            # Get the first row to determine impact type for color
            row = track_works.df[track_works.df['cou_id'] == id].iloc[0]
           
            expander_text = f"🚧 COU ID: {id} ({len(track_works.df[track_works.df['cou_id'] == id])} items)"
            # Create expandable section for each cou_id
            with st.expander(expander_text, expanded=False):
                info_row = track_works.df[track_works.df['cou_id'] == id].iloc[0]
                col1, col2 = st.columns([4, 6])
 
                with col1:
                    st.markdown(f"📆**Dates:** {info_row['date_begin']} **→** {info_row['date_end']} ({info_row['period']})")
                    st.markdown(f"🕒**Time:** {info_row['time_begin']} **→** {info_row['time_end']} ({info_row['period_type']})")
                    st.markdown(f"🛤️**Track:** {info_row['tracks']} ({"secondary" if info_row['is_secondary'] == '1' else 'primary'})")
                    st.markdown(f"📝**Description:** {descriptions.get(id, 'No description available')}")

                with col2:
                    df_filtered = track_works.df[track_works.df['cou_id'] == id]
                    
                    df_display = pd.DataFrame(df_filtered.copy())
                    df_display['section_from_name'] = df_display['section_from_id'].apply(
                        lambda x: oper_points.get_name_by_id(x) if pd.notna(x) else 'N/A'
                    )
                    df_display['section_to_name'] = df_display['section_to_id'].apply(
                        lambda x: oper_points.get_name_by_id(x) if pd.notna(x) else 'N/A'
                    )
                    
                    df_display = pd.DataFrame(df_display[['section_from_name', 'section_to_name', 'impact']].copy())
                    df_display.columns = ['From', 'To', 'Impact Type']
                    st.dataframe(df_display)

                
            