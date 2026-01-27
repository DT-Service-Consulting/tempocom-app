from typing import List
from folium.map import FeatureGroup
from features.operational_points.OperationalPoints import OperationalPoints
from features.operational_points.OperationalPointsView import OperationalPointsView

class OperationalPointsController:
    def __init__(self):
        self.operational_points = OperationalPoints().load()
        self.operational_points_view = OperationalPointsView()

    def get_operational_points_layer(self) -> List[FeatureGroup]:
        return [self.operational_points_view.get_operational_points_layer(
            self.operational_points
        )]