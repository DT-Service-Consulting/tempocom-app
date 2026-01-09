import streamlit as st
from tempocom.modules.streamlit import Page
from features.punctuality.PunctualityController import PunctualityController
from tempocom.services import DBConnector



class PunctualityManagement(Page):
    title = "Punctuality Management"
    layout = "wide"

    @st.cache_resource
    def _get_punctuality_controller(_self, _dbc):
        return PunctualityController(_dbc)

    def render(self):
        punctuality_controller = self._get_punctuality_controller(self.dbc)

        selected_relations = st.multiselect("Select relations you want to compare", punctuality_controller.get_unique_relations())

        if selected_relations:
            st.plotly_chart(punctuality_controller.get_relations_boxplots_chart(selected_relations, is_two_way=True))
            rel_for_stations_selection = st.segmented_control("Select stations performance of", selected_relations, default=selected_relations[0])
            if rel_for_stations_selection:
                st.plotly_chart(punctuality_controller.get_stations_boxplots_chart(rel_for_stations_selection))

if __name__ == "__main__":
    punctuality_management = PunctualityManagement()