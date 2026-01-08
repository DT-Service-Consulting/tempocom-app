import streamlit as st
from streamlit_folium import folium_static
import folium
from components import Responsiveness

class Map:
    def __init__(self, m: folium.Map = None, zoom_start: int = 8):
        self.tiles = "cartodb dark_matter"
        self.zoom_start = zoom_start
        self.m = m if m is not None else folium.Map(location=[50.50, 4.50], zoom_start=self.zoom_start, tiles=self.tiles)
        self.legends = []  # Stores legends of all added layers

    def add_layers(self, layers: list):
        """
        layers: list of layer_feature or (layer_feature, legend_list) tuples.
        - layer_feature: a folium map layer (FeatureGroup, etc)
        - legend_list: a list of legend dicts for that layer
        """
        for layer_info in layers:
            if isinstance(layer_info, tuple) and len(layer_info) == 2:
                layer, legend = layer_info
                if hasattr(layer, 'add_to'):
                    layer.add_to(self.m)
                if isinstance(legend, list):
                    self.legends.extend(legend)
                elif legend is not None:
                    self.legends.append(legend)
            else:
                if hasattr(layer_info, 'add_to'):
                    layer_info.add_to(self.m)
        return self

    def render(self, ratio: float = 0.5):
        r = Responsiveness() 
        folium_static(self.m, width=r[0], height=int(r[1] * ratio))

    def get_legend_html(self, legends: list):
        # legends should be a list of dict legends OR a list of legend descriptors
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: auto; height: auto;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 5px;">
            <p style="margin-top: 0; font-weight: bold;">Legend</p>
        """

        # Support for flexible legend formats
        for item in legends:
            if isinstance(item, dict):
                # MacroLinks: e.g., {"Macro Link": ""}
                # OperationalPoints: e.g., {"name":"Macro Stations", "icon": "circle", ...}
                if "icon" in item and "name" in item:
                    icon = item.get("icon", "")
                    color = item.get("color", "black")
                    name = item.get("name", "")
                    # If name is iterable (like dict_keys), join for display
                    if not isinstance(name, str):
                        name = " / ".join(str(n) for n in name)
                    legend_html += f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <i class="fa fa-{icon}" style="color: {color}; margin-right: 5px;"></i>
                        <span>{name}</span>
                    </div>
                    """
                else:
                    # Default: show each key as label, value as icon
                    for label, icon in item.items():
                        legend_html += f"""
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <i class="fa fa-{icon}" style="color: orange; margin-right: 5px;"></i>
                            <span>{label}</span>
                        </div>
                        """
            elif isinstance(item, str):
                legend_html += f"""<span>{item}</span><br>"""

        legend_html += "</div>"
        return legend_html

    def get_legend_markdown(self):
        pass