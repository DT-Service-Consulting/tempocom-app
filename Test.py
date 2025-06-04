import streamlit as st
import pydeck as pdk
import pandas as pd

data = pd.DataFrame([
    {
        "lat": (48.85 + 48.86) / 2,
        "lon": (2.35 + 2.36) / 2,
        "label": "[!]"
    }
])

text_layer = pdk.Layer(
    "TextLayer",
    data=data,
    get_position='[lon, lat]',
    get_text="label",
    get_color=[255, 255, 0],
    get_size=10,
    get_alignment_baseline="'bottom'",
    get_text_anchor="'middle'",
    billboard=True,
)

view_state = pdk.ViewState(
    latitude=data["lat"].iloc[0],
    longitude=data["lon"].iloc[0],
    zoom=13,
)

deck = pdk.Deck(
    layers=[text_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v9",
)

st.pydeck_chart(deck)