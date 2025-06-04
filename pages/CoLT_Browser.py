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
from components.page_template import page_template

title = "🚧 CoLT Browser"
page_template(title)

@st.cache_data()
def load(): return MacroNetwork(), Coupures(), json.load(open("constants.json"))
network, coupures, constants = load()
if 'network' not in st.session_state: st.session_state.network = network
if 'coupures' not in st.session_state: st.session_state.coupures = coupures
if 'constants' not in st.session_state: st.session_state.constants = constants
if 'filtered_cou_ids' not in st.session_state: st.session_state.filtered_cou_ids = coupures.coupures['cou_id'].unique().tolist()
if 'current_coupure' not in st.session_state: st.session_state.current_coupure = coupures.coupures['cou_id'].unique().tolist()[0]
if 'm' not in st.session_state:
    m = folium.Map(location=[50.850346, 4.351721], zoom_start=8, tiles='CartoDB dark_matter')
    m = network.render_macro_network(m)
    st.session_state.m = m
network = st.session_state.network
coupures = st.session_state.coupures
constants = st.session_state.constants
filtered_cou_ids = st.session_state.filtered_cou_ids
current_coupure = st.session_state.current_coupure
m = st.session_state.m

height, width, ratio = ResponsiveMap()

    

# ------------------------------------------------------------
#                           RENDER
# ------------------------------------------------------------

# Filter form
with st.form("filter_coupure"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        coupure_id = st.text_input("Search coupure by ID 🔢", key="coupure_id")
    with col2:
        st.markdown("<div style='text-align: center;'>OR</div>", unsafe_allow_html=True)
    with col3:
        leader = st.multiselect("Filter by leader 👷‍♂️", options=coupures.leaders, key="leader")
    with col4:
        period = st.multiselect("Filter by period type ☀️", options=coupures.period_type, key="period_type")
    with col5:
        status = st.multiselect("Filter by status ✅", options=coupures.status, key="status")
    search = st.form_submit_button("Search 🔍")

    if search:
        filter = {
            'cou_id': coupure_id,
            'leader': leader,
            'period_type': period,
            'status': status
        }
        filtered_cou_ids = coupures.get_cou_id_list_by_filter(filter)
        current_coupure = filtered_cou_ids[0]
        st.session_state.filtered_cou_ids = filtered_cou_ids
        st.session_state.current_coupure = current_coupure

col1, col2 = st.columns(2)
with col1:
    if st.button("⬅ Previous", use_container_width=True):
        st.session_state.current_coupure = max(min(filtered_cou_ids), filtered_cou_ids[filtered_cou_ids.index(current_coupure) - 1])
with col2:
    if st.button("Next ➡", use_container_width=True):
        st.session_state.current_coupure = min(max(filtered_cou_ids), filtered_cou_ids[filtered_cou_ids.index(current_coupure) + 1])
st.write(f"Displaying the coupure {current_coupure}")

m = folium.Map(location=[50.850346, 4.351721], zoom_start=8, tiles='CartoDB dark_matter')
m = network.render_macro_network(m)
layer = coupures.render_coupure(current_coupure, network)
if layer:
    layer.add_to(m)
else:
    st.warning("Coupure not found in the data.")

folium.LayerControl().add_to(m)
folium_static(m, width=width, height=int(height * ratio))

st.info(f"""
        **Description of the coupure:** 
        {coupures.descriptions[coupures.descriptions['cou_id'] == current_coupure]['description_of_works'].values[0]}
        """)
df = coupures.coupures[coupures.coupures['cou_id'] == current_coupure]
if not df.empty:st.write(df)
else: st.warning("Coupure not found in the data.")



