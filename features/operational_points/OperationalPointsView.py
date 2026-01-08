import folium
import ast
import pandas as pd
from features.operational_points.OperationalPoints import OperationalPoints

class OperationalPointsView:
    icon_size = "5"
    icons = {
            '6': 'question-circle',
            '642': 'question-circle',
            'Block signal with reporting': 'traffic-light',
            'Connection': 'random',
            'Grid': 'th',
            'Junction': 'exchange-alt',
            'Movable bridge': 'bridge',
            'N/A': 'ban',
            'Net border': 'globe-europe',
            'Other': 'question-circle',
            'SAS': 'door-open',
            'Service installation': 'wrench',
            'Service stop': 'stop-circle',
            'Station': 'stop-circle',
            'Stop in open track': 'stop-circle'
        }
        
    def get_macro_stations_layer(self, oper_points:OperationalPoints) -> folium.FeatureGroup:
        layer = folium.FeatureGroup(name="Macro Stations")
        #filter isin_ma
        df = oper_points.df[oper_points.df['ISIN_MACRO_NETWORK'] == True]
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=row['GEO_POINT'],
                radius=1 if row.get('N_LINKS', 0)<=2  else 3,
                color='white',
                fill=True,
                fill_color='#06BCF1',
                fill_opacity=1,
                popup=f"{row['NAME']} ({row.get('N_LINKS', 0)} links)",
                tooltip=f"{row['NAME']} ({row.get('N_LINKS', 0)} links)"
            ).add_to(layer)
        return layer


    def get_operational_points_layer(self, oper_points:OperationalPoints) -> folium.FeatureGroup:
        layer = folium.FeatureGroup(name="Operational Points")
        df = oper_points.df[~oper_points.df['ISIN_MACRO_NETWORK']]
        for _, row in df.iterrows():
            folium.Marker(
                    location=row['GEO_POINT'],
                    icon=folium.DivIcon(
                        html=f"""
                            <div style="
                                background-color: orange;
                                color: white;
                                border-radius: 50%;
                                width: 20px;
                                height: 20px;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                font-size: 12px;
                                border-color: white;
                                border-width: 2px;
                                border-style: solid;
                            ">
                                <i class=\"fa fa-{self.icons.get(row.get('CLASS'), 'question-circle')}\"></i>
                            </div>
                        """,
                        icon_size=(20, 20),
                    ),
                    popup=f"{row.get('CLASS')} ({row['NAME']})",
                    tooltip=f"{row.get('CLASS')} ({row['NAME']})",
                    draggable=False,
                    keyboard=False
                ).add_to(layer)
        return layer
