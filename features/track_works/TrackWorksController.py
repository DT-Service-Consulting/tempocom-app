from features.track_works.TrackWorks import TrackWorks
from features.track_works.TrackWorksView import TrackWorksView
from features.operational_points.OperationalPoints import OperationalPoints
from datetime import datetime
from features.macro_network.MacroLinks import MacroLinks
from typing import List
from folium.map import FeatureGroup
from tempocom.services import DBConnector

class TrackWorksController:
    
    def __init__(self, dbc: DBConnector, macro_network: MacroLinks):

        self.dbc = dbc
        # models
        self.track_works = TrackWorks(dbc)
        self.macro_links = MacroLinks(dbc)
        self.operational_points = OperationalPoints(dbc)
        # views
        self.track_works_view = TrackWorksView()
        # operations

    
    def get_track_works_layers(self, date=None) -> List[FeatureGroup]:
        if date is None:
            date = datetime.now()
        track_works = TrackWorks(self.dbc, date)
        layer = self.track_works_view.get_track_works_layer(
            track_works,
            self.operational_points,
            self.macro_links,
            date
        )
        return [layer]
    
    def get_active_works_count(self) -> int:
        return len(self.track_works.df) if self.track_works.df is not None else 0
    
    def get_works_by_impact(self, impact_type: str):
        if self.track_works.df is None:
            return []
        return self.track_works.df[self.track_works.df['impact'] == impact_type]

    def get_table_view(self):
        return self.track_works_view.get_table_view(self.track_works, self.operational_points)
    
    def get_date_range(self):
        return self.track_works.date_range