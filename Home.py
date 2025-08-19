import streamlit as st
import os
import sys
from components.page_template import page_template
#st.set_page_config(page_title="About - TEMPOCOM", page_icon="🚄", layout="wide")
page_template("🏠 Home")
sys.path.append('../')
readme_path =  "README.md"

if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    readme_content = readme_content.replace('<p align="center">', '<div style="text-align: center;">')
    readme_content = readme_content.replace('</p>', '</div>')
    readme_content = readme_content.replace('<h1 align="center">', '<h1 style="text-align: center;">')
    readme_content = readme_content.replace('<h3 align="center">', '<h3 style="text-align: center;">')
    readme_content = '\n'.join([line for line in readme_content.split('\n') if not line.strip().startswith('![') and not '<img' in line])
    st.markdown(readme_content, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🤝 Partners")
    st.markdown("""
    This project is developed in collaboration with:
    - **Infrabel** - Belgian railway infrastructure manager
    - **MLG** - Machine Learning Group of the University of Brussels
    - **DTSC** - DT Services & Consulting
    """)

    st.markdown("---")
    st.markdown("### 📊 Project Timeline - Gantt Diagram")
    
    import plotly.express as px
    import pandas as pd
    from datetime import datetime
    
    tasks_kpi = [
        ("KPI 1 - Business and End-User Needs Analysis", "2025-04-01", "2025-05-31", "Analysis"),
        ("KPI 2 - R&D Program Dedicated to Algorithms", "2025-05-01", "2026-07-31", "R&D"),
        ("KPI 3 - Architecture and Solution Design", "2025-11-01", "2026-04-30", "Design"),
        ("KPI 4 - Development and Integration", "2026-07-01", "2026-12-31", "Development"),
        ("KPI 5 - Validation and Iteration", "2026-09-01", "2026-12-31", "Validation"),
        ("KPI 6 - Documentation and Deployment Preparation", "2026-11-01", "2026-12-31", "Deployment"),
    ]
    
    colors = {
        "Analysis": "#FF6B6B",
        "Design": "#4ECDC4", 
        "R&D": "#45B7D1",
        "Development": "#96CEB4",
        "Validation": "#FF9FF3",
        "Deployment": "#54A0FF"
    }
    
    df_tasks = pd.DataFrame(tasks_kpi, columns=["Task", "Start", "Finish", "Category"])
    df_tasks["Start"] = pd.to_datetime(df_tasks["Start"])
    df_tasks["Finish"] = pd.to_datetime(df_tasks["Finish"])
    
    fig = px.timeline(df_tasks, x_start="Start", x_end="Finish", y="Task", color="Category",
                      color_discrete_map=colors, title="TEMPOCOM Project Timeline")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Timeline", yaxis_title="Tasks", height=500)
    
    now = datetime.now()
    fig.add_vline(x=now.timestamp() * 1000, line_dash="dash", line_color="red", 
                  annotation_text="NOW", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Fichier README.md non trouvé")

