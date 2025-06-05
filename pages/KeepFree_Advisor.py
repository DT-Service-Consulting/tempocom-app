
from components import *
from objects import Coupures, MacroNetwork
import folium
from streamlit_folium import folium_static
from components.page_template import page_template
import pandas as pd


title = "📈🚉Keep Free Advisor"
page_template(title)
height,width, ratio = ResponsiveMap()

@st.cache_data()
def load(): return Coupures(), MacroNetwork()
coupures, network = load()
if 'coupures' not in st.session_state: st.session_state.coupures = coupures
if 'network' not in st.session_state: st.session_state.network = network
if 'advised_coupures' not in st.session_state: st.session_state.advised_coupures = []
if 'ctl_section' not in st.session_state: st.session_state.ctl_section = None
if 'm' not in st.session_state:
    m = folium.Map(location=[50.850346, 4.351721], zoom_start=8, tiles='CartoDB dark_matter')
    m = network.render_macro_network(m)
    st.session_state.m = m
coupures = st.session_state.coupures
network = st.session_state.network
m = st.session_state.m
advised_coupures = st.session_state.advised_coupures
ctl_section = st.session_state.ctl_section

# ------------------------------------------------------------
#   RENDERING
# ------------------------------------------------------------

with st.form("Keep Free Advisor"):
    col1, col2 = st.columns(2)
    with col1:
        ctl_section = st.selectbox("Section to cut", options = coupures.get_ctl_sections(), index=0)
    with col2:
        submit = st.form_submit_button("Add a coupure")

    if submit:
        st.session_state.ctl_section = ctl_section
        advised_coupures = coupures.advise_keepfrees(ctl_section, network)
        advised_coupures = pd.DataFrame(advised_coupures)
        advised_coupures = advised_coupures
        layer = coupures.render_coupure_line(advised_coupures[advised_coupures['impact'] == 'CTL'], network, opacity=1, line_weight=3, layer_name='Keep Free Advisor')
        advised_coupures = advised_coupures[advised_coupures['impact'] != 'CTL']
        st.session_state.advised_coupures = advised_coupures
        layer.add_to(m)

mcol1,mcol2 = st.columns(2)
with mcol1:
    folium_static(m, width=width, height=int(height * ratio))

with mcol2:
    if 'advised_coupures' in locals():
        with st.form("select_keep_frees"):
            st.subheader("Advised Keep Frees")
            selected_keepfrees = []
            for idx, row in advised_coupures.iterrows():
                st.divider()
                if row['impact'] == 'CTL':
                    continue
                col1, col2, col3, col4 = st.columns([3,3,2,1])
                with col1:
                    st.write(f"From: {row['section_from_name']}")
                with col2:
                    st.write(f"To: {row['section_to_name']}")
                with col3:
                    st.write(f"Score: {row['nb_occ']}")
                with col4:
                    selected = st.checkbox("+", key=f"select_{idx}", label_visibility="hidden")
                    if selected:
                        selected_keepfrees.append(row)
            submit_keepfrees = st.form_submit_button("Apply Keep Frees")

            if submit_keepfrees and selected_keepfrees:
                layer = coupures.render_coupure_line(pd.DataFrame(selected_keepfrees), network, opacity=1, line_weight=3, layer_name='Keep Free Advisor')
                layer.add_to(m)
                st.rerun()

        







