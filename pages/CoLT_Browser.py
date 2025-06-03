#IMPORTS
import streamlit as st
from objects import *
import folium
from streamlit_folium import folium_static
import json
from components.SecureLogin import SecureLogin
from components.ResponsiveMap import ResponsiveMap
from components.LegendColt import LegendColt
import os
import pandas as pd
st.set_page_config(layout="wide", page_title="CoLT Browser", page_icon="assets/favicon.ico")

# ------------------------------------------------------------
#                           SECURITY
# ------------------------------------------------------------
if os.getenv('ENVIRONMENT') == 'production':
    from components.SecureLogin import SecureLogin
    if not SecureLogin().render(): 
        st.stop()

# ------------------------------------------------------------
#                           CACHING
# ------------------------------------------------------------
@st.cache_data()
def load():
    print("Loading data...")
    suggestions_df = pd.read_csv('./mart/private/colt_dat_S1_model.csv')
    return MacroNetwork(), Coupures(), json.load(open("constants.json")), suggestions_df
# ------------------------------------------------------------
#                           INIT
# ------------------------------------------------------------
network, coupures, constants, suggestions_df = load()
height, width, ratio = ResponsiveMap()
m = folium.Map(location=[50.850346, 4.351721], zoom_start=8, tiles='CartoDB dark_matter', width=width, height=height)

status_select = [f"{i} ({constants['colt_status_details'][i]})" for i in constants['colt_status_details'].keys()]

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
            🚧 CoLT Browser
    </h1>''', 
    unsafe_allow_html=True)

# ------------------------------------------------------------
#                           RENDER
# ------------------------------------------------------------



# Initialize session state if not exists
if 'current_coupure_index' not in st.session_state:
    st.session_state.current_coupure_index = 0
    st.session_state.filtered_coupures = coupures.coupures['cou_id'].unique().tolist()


st.markdown("### Browser")
# Filter form
with st.form("filter_coupure"):
    col1, col2, col3, col4,col5 = st.columns(5)
    with col1:
        coupure_id = st.text_input("Search coupure by ID 🔢", key="coupure_id")
    with col2:
        impact = st.multiselect("Filter by impact 🚫", options=['Keep Free', 'CTL', 'SAVU', 'Travaux possibles', 'Autre'], key="impact")
    with col3:
        leader = st.multiselect("Filter by leader 👷‍♂️", options=coupures.coupures['leader'].dropna().sort_values().unique(), key="leader")
    with col4:
        period = st.multiselect("Filter by period type ☀️", options=["Day","Night","Continuous"], key="period_type")
    with col5:
        status = st.multiselect("Filter by status ✅", options=status_select, key="status")
    
    if st.form_submit_button("Search 🔍"):
        filtered_df = coupures.coupures
        if impact:
            if 'Autre' in impact:
                specific_impacts = [i for i in impact if i != 'Autre']
                filtered_df = filtered_df[
                    (filtered_df['impact'].apply(lambda x: all(i in str(x) for i in specific_impacts))) |
                    (~filtered_df['impact'].apply(lambda x: all(i in str(x) for i in ['Keep Free', 'CTL', 'SAVU', 'Travaux possibles'])))
                ]
            else:
                filtered_df = filtered_df[filtered_df['impact'].apply(lambda x: all(i in str(x) for i in impact))]
        if leader:
            filtered_df = filtered_df[filtered_df['leader'].isin(leader)]
        if period:
            filtered_df = filtered_df[filtered_df['period_type'].isin(period)]
        if status:
            filtered_df = filtered_df[filtered_df['status'].isin([i.split(" ")[0] for i in status])]
        st.session_state.filtered_coupures = filtered_df['cou_id'].unique().tolist()
        if st.session_state.current_coupure_index >= len(st.session_state.filtered_coupures):
            st.session_state.current_coupure_index = 0
        if coupure_id:
            try:
                coupure_id = int(coupure_id)
                if coupure_id in coupures.coupures['cou_id'].values:
                    if coupure_id in st.session_state.filtered_coupures:
                        idx = st.session_state.filtered_coupures.index(coupure_id)
                        st.session_state.current_coupure_index = idx
                    else:
                        # Find closest higher coupure ID in filtered list
                        higher_ids = [id for id in st.session_state.filtered_coupures if id > coupure_id]
                        if higher_ids:
                            closest_id = min(higher_ids)
                            idx = st.session_state.filtered_coupures.index(closest_id)
                            st.session_state.current_coupure_index = idx
                            st.warning(f"Showing closest available coupure: {closest_id}")
                        else:
                            st.error("No matching coupures found after filtering")
                else:
                    st.error("Coupure ID not found")
            except ValueError:
                st.error("Please enter a valid ID")
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ Previous",use_container_width=True):
        if st.session_state.current_coupure_index > 0:
            st.session_state.current_coupure_index -= 1
with col2:
    if st.button("Next ➡",use_container_width=True):
        if st.session_state.current_coupure_index < len(st.session_state.filtered_coupures) - 1:
            st.session_state.current_coupure_index += 1


# Display current coupure
if 0 <= st.session_state.current_coupure_index < len(st.session_state.filtered_coupures):
    current_coupure = st.session_state.filtered_coupures[st.session_state.current_coupure_index]
    st.write(f"Showing coupure {current_coupure} (total: {len(st.session_state.filtered_coupures)})")

    m = network.render_macro_network(m)
    layer = coupures.render_coupure(current_coupure, network)
    if layer:
        layer.add_to(m)
    else:
        st.warning("Coupure non trouvée dans les données.")
    col1, col2 = st.columns([3,1])
    with col1:
        folium.LayerControl().add_to(m)
        folium_static(m)
    with col2:
        st.markdown("### Legend")
        st.markdown(LegendColt(coupures.PALETTES), unsafe_allow_html=True)

    df = coupures.coupures[coupures.coupures['cou_id'] == current_coupure]
    if not df.empty:
        st.write(df)
    else:
        st.warning("Coupure non trouvée dans les données.")
else:
    st.error("Invalid coupure index")



