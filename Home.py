import pretty_errors
import streamlit as st
from components import *

# ------------------------------------------------------------
#                           LABS
# ------------------------------------------------------------
labs = [
    {
        "title": "✎ Network Editor",
        "description": "Modify links between stations and calculate the shortest path between two stations.",
        "image": "assets/network_icon.png",
        "internal": False,
        "redirect": "pages/Network_Editor.py",
        "available": True
    },
    {
        "title": "🖥️Railway Monitor",
        "description": "Monitor the railway network. See the status of the ETCS, the incidents and many other things.",
        "image": "assets/simulation_icon.png",
        "internal": False,
        "redirect": "pages/Railway_Monitor.py",
        "available": True
    },
    {
        "title": "🚧CoLT Advisor",
        "description": "Get the best advises on Keeping free lines when a line is closed. Infrabel Use Case.",
        "image": "assets/maintenance_icon.png",
        "internal": True,
        "redirect": "pages/CoLT_Advisor.py",
        "available": True
    },
    {
        "title": "🌊Domino Effect",
        "description": "Analyze the domino effect of disruptions on the railway network.",
        "image": "assets/maintenance_icon.png",
        "internal": True,
        "redirect": "pages/Domino_Effect_Analyzer.py",
        "available": False
    }
]

# ------------------------------------------------------------
#                           PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="TEMPOCOM - Railway Digital Twin", page_icon="🚄")
st.logo("assets/logo.png",size="large")

with st.sidebar:
    st.markdown("Data provided by")
    st.image("assets/infrabel.png", width=200, clamp=True)
    st.markdown("Developed by")
    st.image("assets/brain-logo.png", width=200, clamp=True)
    
st.markdown(
    '''<h1 style='text-align: center;'>
            🔬All Labs🥼
    </h1>''', unsafe_allow_html=True)

for i in range(0, len(labs), 3):
    cols = st.columns(3)
    for col, lab in zip(cols, labs[i:i+3]):
        with col:
            st.markdown(labs_card(lab), unsafe_allow_html=True)
            if st.button("COMING 🔜" if not lab.get('available', True) else "↳ Access to lab 🧪", 
                     key=lab['title'], 
                     disabled=not lab.get('available', True),
                     help="This lab is not yet available" if not lab.get('available', True) else None):
                st.switch_page(lab['redirect'])

st.markdown(arrow_element(),unsafe_allow_html=True)
st.markdown(digital_twin_card(), unsafe_allow_html=True)
st.markdown(end_page(), unsafe_allow_html=True)
