import streamlit as st
from tempocom.modules.streamlit import Page
from streamlit_option_menu import option_menu
from features.macro_network.MacroNetworkController import MacroNetwork
from features.track_works.TrackWorksController import TrackWorksController
from features.track_works.TrackWorks import TrackWorks
from components.map.Map import Map
from datetime import datetime

class TrackWorksAdvisor(Page):
    private = True
    
    @st.cache_resource
    def _get_macro_network(_self) -> MacroNetwork:
        return MacroNetwork()
    
    @st.cache_resource
    def _get_track_works_controller(_self) -> TrackWorksController:
        return TrackWorksController()

    def render(self):
        menu = option_menu(None,
            ["Current Track Works", 
            "CoLT Advisor"
            ],orientation="horizontal"
            )

        if menu == "Current Track Works":
            # ┌─────────────────────────────────────────────────────────────┐
            # │                    Current Track Works                      │
            # └─────────────────────────────────────────────────────────────┘


            macro_network = self._get_macro_network()
            track_works_controller = self._get_track_works_controller()

            min_date, max_date = track_works_controller.get_date_range()
            
            if min_date and max_date:
                today = datetime.now().date()
                default_date = min(max(today, min_date), max_date)
                
                with st.form(key="track_works_form"):
                    selected_date = st.slider(
                        "Select date to view track works",
                        min_value=min_date,
                        max_value=max_date,
                        value=default_date,
                        format="DD/MM/YYYY"
                    )
                    
                    manual_date = st.date_input(
                        "Or enter date manually",
                        value=default_date,
                        min_value=min_date,
                        max_value=max_date,
                        format="DD/MM/YYYY"
                    )
                    if manual_date != selected_date:
                        selected_date = manual_date
                    
                    st.form_submit_button("Submit")
                
                selected_datetime = datetime.combine(selected_date, datetime.min.time())
                
                track_works_controller_filtered = TrackWorksController()
                track_works_controller_filtered.track_works = TrackWorks().load(selected_datetime)
            else:
                st.warning("No track works data available")
                return

            cols = st.columns(2)
            with cols[0]:
                layers = []
                layers.extend(macro_network.get_macro_links_layer())
                layers.extend(macro_network.get_macro_stations_layer())
                layers.extend(track_works_controller_filtered.get_track_works_layers(selected_datetime))
                Map(zoom_start=7).add_layers(layers).render(ratio=0.5)    
            
            with cols[1]:
                with st.container(height=600):
                    st.dataframe(track_works_controller_filtered.get_table_view())

        elif menu == "CoLT Advisor":
            # ┌─────────────────────────────────────────────────────────────┐
            # │                    CoLT Advisor                             │
            # └─────────────────────────────────────────────────────────────┘

            st.write("CoLT Advisor")

if __name__ == "__main__":
    track_works_advisor = TrackWorksAdvisor()