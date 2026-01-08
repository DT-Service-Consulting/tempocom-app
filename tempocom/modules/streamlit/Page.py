import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import inspect
from app.components import NavBar
from app.styles import load_global_styles, load_theme
from tempocom.services import DBConnector

class Page:
    """
    Template de base pour les pages Streamlit du projet TEMPOCOM.

    - `__init__` configure automatiquement la page et exécute `render()` de l'enfant.
    - Un footer est ajouté automatiquement en bas de chaque page.
    """

    title = "TEMPOCOM"
    icon = f"{os.getenv('ASSETS_PATH')}/favicon.ico"
    layout = "wide"
    theme = "light"
    private = False

    def __init__(self):
        # Configure la page
        st.set_page_config(page_title=self.title, page_icon=self.icon, layout=self.layout)
        
        # Vérification de l'authentification pour les pages privées
        if self.private and not st.session_state.get("login", False):
            st.session_state.redirect_after_login = st.session_state.get("current_page", "Home.py")
            st.switch_page("pages/Login.py")
            return
        
        # Charge les styles globaux AVANT la navbar
        self._load_styles()
        
        NavBar()
        st.divider()
        # Appelle automatique du render() de la classe héritée
        child_render = getattr(self, "render", None)
        if child_render and callable(child_render):
            # Vérifie que ce n'est pas la méthode par défaut
            if not self._is_abstract_render(child_render):
                child_render()
                self._render_footer()

    def _load_styles(self):
        """Charge les styles CSS globaux et applique le thème"""
        global_css = load_global_styles()
        theme_script = load_theme("light")
        
        st.markdown(f"<style>{global_css}</style>", unsafe_allow_html=True)
        st.markdown(theme_script, unsafe_allow_html=True)

    def _is_abstract_render(self, method):
        return inspect.getsource(method).strip().startswith("raise NotImplementedError")

    def _render_footer(self):
        """Footer professionnel avec informations légales et crédits"""
        
        # Définir le chemin des assets
        assets_path = os.getenv('ASSETS_PATH')

        # Footer avec largeur contrôlée et contenu en anglais
        footer_html = f"""
        <style>
        .footer-container {{
            background-color: #1a1a1a;
            color: white;
            padding: 40px 20px;
            margin: 50px 0 0 0;
            border-radius: 10px;
            max-width: 100%;
            overflow: hidden;
        }}
        .footer-title {{
            color: #ffffff;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }}
        .footer-text {{
            color: #cccccc;
            line-height: 1.7;
            margin-bottom: 20px;
        }}
        .logo-img {{
            height: 40px;
            margin-right: 15px;
            vertical-align: middle;
            background-color: white;
            padding: 8px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .badge {{
            background-color: #333333;
            color: #ffffff;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
            margin: 8px 8px 8px 0;
            font-weight: bold;
        }}
        .badge-success {{ background-color: #28a745; }}
        .badge-info {{ background-color: #17a2b8; }}
        .badge-warning {{ background-color: #fd7e14; }}
        .copyright {{
            border-top: 1px solid #333333;
            padding-top: 25px;
            margin-top: 40px;
            color: #999999;
            font-size: 0.95em;
            text-align: center;
        }}
        .tech-stack {{
            color: #bbbbbb;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        </style>
        
        <table class="footer-container" width="100%" cellpadding="25" cellspacing="0">
        <tr>
            <td width="33%" valign="top">
                <p class="footer-title">
                    <img src="{assets_path}/logo.png" class="logo-img" alt="TEMPOCOM">
                    TEMPOCOM
                </p>
                <p class="footer-text">
                    TEMPOCOM is a Digital Twin developed by BRAIN to optimize railway transport management and identify network capacity losses to enhance robustness and flexibility.
                     <p class="footer-text">
                        <strong>Key Features</strong><br>
                        🔍 <strong>Causal Analysis</strong> of network disruptions<br>
                        🔁 <strong>Data-Driven Simulations</strong> for scenario testing<br>
                        🚦 <strong>Optimization Engine</strong> for strategic decisions<br>
                        🎯 Focus on performance, cost-efficiency & sustainability
                    </p>
                </p>
                <span class="badge badge-success">🔧 Beta Version</span>
            </td>
            <td width="33%" valign="top">
                <p class="footer-title">
                    <img src="{assets_path}/brain-logo.png" class="logo-img" alt="BRAIN">
                    BRAIN
                </p>
                    <p class="footer-text">
                        <strong>BRAIN</strong> is an AI research center that bridges the gap between academia and the industrial world.<br><br>
                        🎓 <strong>Academic Excellence</strong> & cutting-edge research<br>
                        🏭 <strong>Industrial Applications</strong> & real-world solutions<br>
                        🤝 <strong>Knowledge Transfer</strong> from research to practice<br>
                        🔬 <strong>Innovation Hub</strong> for AI-driven technologies
                    </p>
                <span class="badge badge-info">💡 Innovation</span>
            </td>
            <td width="33%" valign="top">
                <p class="footer-title">
                    <img src="{assets_path}/infrabel.png" class="logo-img" alt="INFRABEL">
                    Data & Legal
                </p>
                <p class="footer-text">
                    <strong>⚠️ Non-Disclosure Agreement (NDA)</strong><br><br>
                    Data proprietary to Infrabel<br>
                    Partially internal use only (Private & Public data)
                </p>
                <span class="badge badge-warning">🔒 Confidential</span>
            </td>
        </tr>
        <tr>
            <td colspan="3">
                <p class="copyright">
                    © 2025 BRAIN - All rights reserved | Infrabel data under confidentiality agreement<br>
                    Powered by Python • Streamlit • Folium
                </p>
            </td>
        </tr>
        </table>
        """
        
        st.markdown(footer_html, unsafe_allow_html=True)
