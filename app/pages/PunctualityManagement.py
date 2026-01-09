import streamlit as st
from tempocom.modules.streamlit import Page
from features.punctuality.PunctualityController import PunctualityController
from tempocom.services.DBConnector import get_db_connector



class PunctualityManagement(Page):
    title = "Punctuality Management"
    layout = "wide"

    @st.cache_resource
    def _get_db_connector(_self):
        return get_db_connector()
    
    @st.cache_resource
    def _get_punctuality_controller(_self, _dbc):
        return PunctualityController(_dbc)

    def render(self):
        try:
            dbc = self._get_db_connector()
            punctuality_controller = self._get_punctuality_controller(dbc)

            # Try to get unique relations with error handling
            try:
                unique_relations = punctuality_controller.get_unique_relations()
            except Exception as e:
                st.error("❌ Database connection error occurred. Please try refreshing the page.")
                st.error(f"Error details: {str(e)}")
                if st.button("🔄 Retry Connection"):
                    st.cache_resource.clear()
                    st.rerun()
                return

            selected_relations = st.multiselect("Select relations you want to compare", unique_relations)

            if selected_relations:
                try:
                    st.plotly_chart(punctuality_controller.get_relations_boxplots_chart(selected_relations, is_two_way=True))
                    rel_for_stations_selection = st.segmented_control("Select stations performance of", selected_relations, default=selected_relations[0])
                    if rel_for_stations_selection:
                        st.plotly_chart(punctuality_controller.get_stations_boxplots_chart(rel_for_stations_selection))
                except Exception as e:
                    st.error("❌ Error loading chart data. Please try again.")
                    st.error(f"Error details: {str(e)}")
                    if st.button("🔄 Retry"):
                        st.rerun()
        
        except Exception as e:
            st.error("❌ Application error occurred.")
            st.error(f"Error details: {str(e)}")
            if st.button("🔄 Restart Application"):
                st.cache_resource.clear()
                st.rerun()

if __name__ == "__main__":
    punctuality_management = PunctualityManagement()