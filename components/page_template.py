import streamlit as st
from components.SecureLogin import SecureLogin

def page_template(title: str):
    if title != "🔬All Labs🥼":
        if not SecureLogin().render(title): st.stop()
    st.set_page_config(layout="wide", page_title=title, page_icon="assets/favicon.ico")
    st.logo("assets/logo.png",size="large")
    with st.sidebar:
        st.markdown("Data provided by")
        st.image("assets/infrabel.png", width=200, clamp=True)
        st.markdown("Developed by")
        st.image("assets/brain-logo.png", width=200, clamp=True)
        st.link_button("🌐 DTSC-BRAIN", "https://www.brain.dtsc.be/")
        st.link_button("</> Github Repo", "https://github.com/DT-Service-Consulting/tempocom-app")

    st.markdown(
        f'''<h1 style='text-align: center;'>
                {title}
        </h1>''', 
        unsafe_allow_html=True)
    
    return