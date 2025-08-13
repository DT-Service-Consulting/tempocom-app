import streamlit as st
from components.SecureLogin import SecureLogin

def page_template(title: str):
    
    st.set_page_config(layout="wide", page_title=title, page_icon="assets/favicon.ico")
    st.logo("assets/logo.png",size="large")
    hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    if title != "🔬All Labs🥼" and title != "🏠 Home":
        if not SecureLogin().render(title): st.stop()

    with st.sidebar:

        st.page_link("Home.py", label="🏠 Home")
        st.page_link("pages/All_Labs.py", label="🔬All Labs🥼")
        st.page_link("https://brain.dtsc.be/", label="👥 About Us")
        st.page_link("https://github.com/DT-Service-Consulting/tempocom-app", label="📁 Github")
        st.divider()
        st.markdown("Data provided under NDA by")
        st.image("assets/infrabel.png", width=200, clamp=True)
        st.markdown("Developed by")
        st.image("assets/brain-logo.png", width=200, clamp=True)
       
       
    st.markdown(
        f'''<h1 style='text-align: center;'>
                {title}
        </h1>''', 
        unsafe_allow_html=True)
    
    return