from tempocom.modules.streamlit import Page
import streamlit as st
import sys
import os
import re

class About(Page):
    def render(self):
        
        with open("app/content/intro.md", "r", encoding="utf-8") as f:
            content = f.read()
        
            assets_path = os.getenv("ASSETS_PATH")
            if assets_path:
                content = re.sub(r'src="(./assets/[^"]+)"', lambda m: f'src="{assets_path}{m.group(1).replace("./assets", "")}"', content)
                content = re.sub(r'src="(assets/[^"]+)"', lambda m: f'src="{assets_path}{m.group(1).replace("assets", "")}"', content)
            
            st.markdown(content, unsafe_allow_html=True)

if __name__ == "__main__":
    About()