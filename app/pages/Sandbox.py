import streamlit as st
from tempocom.modules.streamlit import Page
from streamlit_option_menu import option_menu
from tempocom.services import DBConnector
from features.macro_network.MacroNetworkController import MacroNetwork
from components.map.Map import Map

class Sandbox(Page):
    title = "Sandbox"
    layout = "wide"


    
    @st.cache_resource
    def _get_macro_network(_self) -> MacroNetwork:
        return MacroNetwork()

    def render(self):
        menu = option_menu(None,
            ["Macro Network Editor", 
            "MicroNetwork Editor"
            ],orientation="horizontal"
            )

        if menu == "Macro Network Editor":
            # ┌─────────────────────────────────────────────────────────────┐
            # │                    Macro Network Editor                     │
            # └─────────────────────────────────────────────────────────────┘

            network = self._get_macro_network()

            with st.form(key='shortest_path_form'):
                col1, col2 = st.columns(2)
                with col1: 
                    station_names = network.operational_points.get_station_names()
                    depart = st.selectbox("Departure station :", station_names, index=None)
                with col2: 
                    arrivee = st.selectbox("Arrival station :", station_names, index=None)
                submit_button = st.form_submit_button("Find shortest path")
                
                if submit_button and depart and arrivee:
                    st.info('The shortest path is found with the Floyd-Warshall algorithm, finding the shortest path between any two stations.')
                    try:
                        dep_id = network.operational_points.get_id_by_name(depart)
                        arr_id = network.operational_points.get_id_by_name(arrivee)
                        
                        if dep_id is not None and arr_id is not None:
                            path_coords = network.macro_links.get_shortest_path_coordinates(dep_id, arr_id)
                            distance = network.macro_links.get_shortest_path_distance(dep_id, arr_id)
                            
                            if path_coords and distance is not None:
                                st.success(f"Shortest path found with a distance of {round(distance, 2)} km")
                                st.session_state.shortest_path = path_coords
                            else:
                                st.error("No path found between the selected stations.")
                                st.session_state.shortest_path = None
                        else:
                            st.error("Invalid station selection.")
                            st.session_state.shortest_path = None
                    except Exception as e:
                        st.error(f"Error calculating path: {str(e)}")
                        st.session_state.shortest_path = None

            st.subheader("Close connections")

            with st.form(key='close_connections_form'):
                available_links = network.macro_links.get_link_names()
                selected_links = st.multiselect(
                    "Select which links to close:", 
                    available_links, 
                    help="Select the connections you want to close. The connections will be closed and the Shortest Path will be calculated depending on the new network."
                )
                submit_button = st.form_submit_button("Close selected links")
                
                if submit_button:
                    if selected_links:
                        closed_links = []
                        for link in selected_links:
                            parts = link.split(" ⇔ ")
                            if len(parts) == 2:
                                dep_id = network.operational_points.get_id_by_name(parts[0])
                                arr_id = network.operational_points.get_id_by_name(parts[1])
                                if dep_id is not None and arr_id is not None:
                                    closed_links.append((dep_id, arr_id))
                        
                        close_links_layer = network.close_links(closed_links)
                        st.session_state.closed_links = closed_links
                        st.success(f"Closed {len(closed_links)} connections")
                    else:
                        st.session_state.closed_links = []

            layers = []
            layers.extend(network.get_macro_links_layer())
            layers.extend(network.get_macro_stations_layer())
            if hasattr(st.session_state, 'closed_links') and st.session_state.closed_links:
                close_links_layer = network.close_links(st.session_state.closed_links)
                layers.append(close_links_layer)
            if hasattr(st.session_state, 'shortest_path') and st.session_state.shortest_path:
                import folium
                path_layer = folium.FeatureGroup(name="Shortest Path")
                folium.PolyLine(
                    locations=st.session_state.shortest_path,
                    color='blue',
                    weight=5,
                    popup="Shortest Path"
                ).add_to(path_layer)
                layers.append(path_layer)

            if hasattr(st.session_state, 'closed_links') and st.session_state.closed_links:
                closed_layer = network.close_links(st.session_state.closed_links)
                layers.append(closed_layer)

            Map(zoom_start=8).add_layers(layers).render()

            
        elif menu == "Micro Network Editor":
            st.write("Micro Network Editor")

if __name__ == "__main__":
    sandbox = Sandbox()