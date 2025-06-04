import pretty_errors
import streamlit as st
from components import *
from components.page_template import page_template
import json

labs = json.load(open("constants.json"))['labs'] 
page_template("🔬All Labs🥼")
  
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
