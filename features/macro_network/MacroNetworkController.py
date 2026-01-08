from features.operational_points.OperationalPoints import OperationalPoints
from features.macro_network.MacroLinks import MacroLinks
from features.operational_points.OperationalPointsView import OperationalPointsView
from features.operational_points.OperationalPointsController import OperationalPointsController
from features.macro_network.MacroNetworkView import MacroNetworkView
from typing import List
from folium.map import FeatureGroup

from tempocom.services import DBConnector  

class MacroNetwork:
    
    def __init__(self, dbc:DBConnector, lang = "EN"):
        # models
        self.operational_points = OperationalPoints(dbc, lang)
        self.macro_links = MacroLinks(dbc, lang)

        # views
        self.operational_points_view = OperationalPointsView()
        self.macro_network_view = MacroNetworkView()

        # operations
        self.operational_points = self.macro_links.update_operational_points_with_links_info(
            self.operational_points
        )

    def get_macro_links_layer(self) -> List[FeatureGroup]:
        return [self.macro_network_view.get_macro_links_layer(
            self.macro_links,
            self.operational_points
        )]
    
    def get_operational_points_layer(self) -> List[FeatureGroup]:
        return [self.operational_points_view.get_operational_points_layer(
            self.operational_points
        )]
    
    def get_macro_stations_layer(self) -> List[FeatureGroup]:
        return [self.operational_points_view.get_macro_stations_layer(
            self.operational_points
        )]

    def close_links(self, closed_links: list[tuple]):
        self.macro_links.close_links(closed_links)
        return self.macro_network_view.get_disabled_macro_links_layer(
            self.macro_links.df.loc[self.macro_links.df['DISABLED']==True],
            self.operational_points
        )

