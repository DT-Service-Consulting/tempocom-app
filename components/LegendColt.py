def LegendColt(palettes):
    html_code = ""
    for impact, palette in palettes.items():
        if palette['dash']:
            style = f"background-color: transparent; border: 3px dashed {palette['color']};"
        else:
            style = f"background-color: {palette['color']}; border: 3px solid {palette['color']};"
        html_code += f"""
            <div style="display: flex; align-items: center; margin-right: 24px;">
                <div style="{style} width: 20px; height: 20px; margin-right: 8px;"></div>
                <span>{palette['label']}</span>
            </div>
        """
    return html_code