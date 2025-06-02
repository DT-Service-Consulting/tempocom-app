import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import folium_static

def ResponsiveMap():
    components.html("""
    <script>
        const getDimensions = () => {
            const dimensions = {
                width: window.innerWidth,
                height: window.innerHeight
            };
            const json = JSON.stringify(dimensions);
            window.parent.postMessage({type: 'dimensions', json: json}, '*');
        };
        window.addEventListener('load', getDimensions);
        window.addEventListener('resize', getDimensions);
    </script>
    """, height=0)

    st.markdown("""
    <script>
        window.addEventListener("message", (event) => {
            if (event.data.type === "dimensions") {
                const dims = JSON.parse(event.data.json);
                const input = document.createElement("input");
                input.type = "hidden";
                input.name = "window_dims";
                input.value = JSON.stringify(dims);
                document.forms[0].appendChild(input);
            }
        });
    </script>
    """, unsafe_allow_html=True)

    width = st.session_state.get("window_width", 1000)
    height = st.session_state.get("window_height", 2000)
    return width, height, 0.7

    # folium_static(map_object, width=width, height=int(height * 0.7)) 