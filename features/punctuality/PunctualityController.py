from features.punctuality.models.Boxplots import Boxplots
from tempocom.services import DBConnector
from features.punctuality.PunctualityView import PunctualityView

class PunctualityController:
    def __init__(self):
        self.boxplots = Boxplots().load()
        self.punctuality_view = PunctualityView()

    def get_unique_relations(self):
        return self.boxplots.get_unique_relations()
    
    def get_relations_boxplots_chart(self, relations: list[str], is_two_way: bool = False):
        return self.punctuality_view.get_boxplots_chart(self.boxplots.get_boxplots_by_relations(relations, is_two_way))

    def get_stations_boxplots_chart(self, relation: str):
        stations = self.boxplots.get_stations_from_relation(relation)
        print(stations)
        return self.punctuality_view.get_boxplots_chart(self.boxplots.get_boxplots_by_stations(stations))