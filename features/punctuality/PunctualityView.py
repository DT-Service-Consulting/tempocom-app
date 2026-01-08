import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class PunctualityView:
    def __init__(self):
        pass

    def get_boxplots_chart(self, boxplots_df):
        df = boxplots_df
        fig = go.Figure()
        
        for i, (_, r) in enumerate(df.iterrows()):
            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            
            # Pre-calculated box
            fig.add_trace(
                go.Box(
                    name=r["name"],
                    x=[r["name"]],
                    q1=[r["q1"]/60],
                    median=[r["median"]/60],
                    q3=[r["q3"]/60],
                    lowerfence=[r["min"]/60],
                    upperfence=[r["max"]/60],
                    boxpoints=False,           # handle outliers separately
                    marker_color=color
                )
            )
        
        fig.update_layout(
            title="Distribution of Delays",
            yaxis_title="Duration (minutes)",
            showlegend=False
        )
        
        return fig