from pathlib import Path

def load_global_styles():
    """Charge tous les styles CSS globaux"""
    styles_dir = Path(__file__).parent
    
    css_files = [
        "global.css",
        "themes.css", 
        "components.css"
    ]
    
    combined_css = ""
    for css_file in css_files:
        css_path = styles_dir / css_file
        if css_path.exists():
            combined_css += css_path.read_text(encoding="utf-8") + "\n"
    
    return combined_css

def load_theme(theme_name="light"):
    """Applique un thème spécifique"""
    return f'<script>document.documentElement.setAttribute("data-theme", "{theme_name}");</script>'
