#IMPORTS
import streamlit as st
import folium
from objects.MacroNetwork import MacroNetwork
from streamlit_folium import folium_static
import pandas as pd
from components.ResponsiveMap import ResponsiveMap
st.set_page_config(layout="wide", page_title="Network Editor", page_icon="assets/favicon.ico")

# ------------------------------------------------------------
#                           CACHING
# ------------------------------------------------------------
@st.cache_data()
def load():
    print("Loading data...")
    return MacroNetwork()

# ------------------------------------------------------------
#                           INIT
# ------------------------------------------------------------
if 'network' not in st.session_state:
    st.session_state.network = load()
network = st.session_state.network

height, width, ratio = ResponsiveMap()
m = folium.Map(location=[50.850346, 4.351721], zoom_start=8, tiles='CartoDB dark_matter', width=width, height=height)
m = network.render_macro_network(m)

# ------------------------------------------------------------
#                           PATTERN
# ------------------------------------------------------------

st.logo("assets/logo.png",size="large")
with st.sidebar:
    st.markdown("Data provided by")
    st.image("assets/infrabel.png", width=200, clamp=True)
    st.markdown("Developed by")
    st.image("assets/brain-logo.png", width=200, clamp=True)


st.markdown(
    '''<h1 style='text-align: center;'>
            ✎ Network Editor
    </h1>''', 
    unsafe_allow_html=True)

# ------------------------------------------------------------
#                           RENDER
# ------------------------------------------------------------

with st.form(key='shortest_path_form'):
    col1, col2 = st.columns(2)
    with col1: depart = st.selectbox("Departure station :", network.stations['Name_FR'].tolist(), index=None)
    with col2: arrivee = st.selectbox("Arrival station :", network.stations['Name_FR'].tolist(), index=None)
    submit_button = st.form_submit_button("Find shortest path")
    
    if submit_button:
        st.info('The shortest path is found with the Floyd-Warshall algorithm, finding the shortest path between any two stations.')
        print("streamlit render_shortest_path")
        print(network.links[network.links['disabled'] == 1])
        m, total_distance, path = network.render_shortest_path(depart, arrivee, m)
        if path:
            st.success(f"The shortest path is found with a distance of {round(total_distance,2)} km.")
        else:
            st.error("No path found between the selected stations.")

st.subheader("Close connections")

with st.form(key='close_connections_form'):
    selected_open = st.multiselect("Select which links to close:", network.get_open_links(), help="Select the connections you want to close. The connections will be closed by cutting the line between the two stations and the Shortest Path will be calculated depending on the new network.")
    submit_button = st.form_submit_button("Close selected links")
    selected_open = [tuple(link.split(" ⇔ ")) for link in selected_open]
   
    m = network.close_links(selected_open, m)
    print("streamlit close_links")
    print(network.links[network.links['disabled'] == 1])

folium.LayerControl().add_to(m)

folium_static(m, width=width, height=int(height * ratio))

