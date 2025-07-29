import streamlit as st
from streamlit_folium import st_folium
from objects.Delay_network import DelayBubbleMap, DelayBubbleMap2 ,DelayLineChart
from objects.Delay_network import DelayHeatmap
from components import *

title = "🌊Domino Effect Analyzer"
page_template(title)

