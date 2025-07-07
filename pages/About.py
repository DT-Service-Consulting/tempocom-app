import streamlit as st
import os
import sys

st.set_page_config(page_title="About - TEMPOCOM", page_icon="🚄", layout="wide")
sys.path.append('../')
readme_path =  "README.md"

if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    readme_content = readme_content.replace('<p align="center">', '<div style="text-align: center;">')
    readme_content = readme_content.replace('</p>', '</div>')
    readme_content = readme_content.replace('<h1 align="center">', '<h1 style="text-align: center;">')
    readme_content = readme_content.replace('<h3 align="center">', '<h3 style="text-align: center;">')
    readme_content = '\n'.join([line for line in readme_content.split('\n') if not line.strip().startswith('![') and not '<img' in line])
    st.markdown(readme_content, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🤝 Partners")
    st.markdown("""
    This project is developed in collaboration with:
    - **Infrabel** - Belgian railway infrastructure manager
    - **MLG** - Machine Learning Group of the University of Brussels
    - **DTSC** - DT Services & Consulting
    """)
else:
    st.error("Fichier README.md non trouvé")

