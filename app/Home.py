import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv; dotenv.load_dotenv('./.env')
import pretty_errors
import streamlit as st
from components import NavBar
from tempocom.modules.streamlit import Page, useState
from components.map.Map import Map
import folium
from features.macro_network.MacroNetworkController import MacroNetwork
from tempocom.services import DBConnector
from features.track_works.TrackWorksController import TrackWorksController
from components.app_card.app_card import app_card



class Home(Page):
    title = "Home"
    layout = "wide"

    
    @st.cache_resource
    def _get_macro_network(_self) -> MacroNetwork:
        return MacroNetwork()
    
    @st.cache_resource
    def _get_track_works_controller(_self) -> TrackWorksController:
        return TrackWorksController()
    
    
    def render(self):
        if st.session_state.get("snow", False):
            st.snow()
            st.session_state.snow = True
        
        
        macro_network = self._get_macro_network()
        map_col, control_col = st.columns([5, 5])

        with control_col:
            st.subheader("")
            st.subheader("Welcome to TEMPOCOM")
            st.text("TEMPOCOM is a digital twin developed by BRAIN to optimize railway transport management. It is a tool that allows you to visualize the network and manage track works.")
            operational_points_check = st.checkbox("Show operational points", key="operational_points_check")

        with map_col:       
            layers = []
            layers.extend(macro_network.get_macro_links_layer())
            layers.extend(macro_network.get_macro_stations_layer())
            if operational_points_check:
                layers.extend(macro_network.get_operational_points_layer())
            Map(zoom_start=7).add_layers(layers).render(ratio=0.4)    
      
                
        # Center the "Apps" section
        st.markdown("""
            <div style="text-align: center; margin: 20px 0;">
                <h2>Apps</h2>
            </div>
        """, unsafe_allow_html=True)
        apps = [
            {
                "title": "Track Works Advisor",
                "subtitle": "Manage track works and their impact on the network",
                "image_path": "traintravaux.jpg",
                "private": True,
                "redirect": "pages/TrackWorksAdvisor.py",
                "available": True
            },
            {
                "title": "Punctuality Management",
                "subtitle": "Manage delays and their impact on the network",
                "image_path": "punctuality.png",
                "private": False,
                "redirect": "pages/PunctualityManagement.py",
                "available": True
            },
            {
                "title": "Traffic Simulator",
                "subtitle": "Simulate traffic and their impact on the network",
                "image_path": "trafficsimulation.png",
                "private": True,
                "redirect": "pages/About.py",
                "available": False
            },
            {
                "title": "Sandbox",
                "subtitle": "Play with the data and the models",
                "image_path": "sandbox.png",
                "private": False,
                "redirect": "pages/Sandbox.py",
                "available": False
            }
        ]
        apps = sorted(apps, key=lambda x: (not x["available"], x["private"]))
        for i in range(0, len(apps), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(apps):
                    with cols[j]:
                        app_index = i + j
                        app = apps[app_index]
                        app_card(app)


if __name__ == "__main__":
    home = Home()