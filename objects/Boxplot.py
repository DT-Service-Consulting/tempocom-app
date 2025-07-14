import pandas as pd
import plotly.express as px

class DelayBoxPlot:
    """
    Creates boxplots of train delays (arrival or departure) by direction.
    """

    def __init__(self, delay_data_path: str):
        """
        Initialize DelayBoxPlot with delay data CSV.

        Args:
            delay_data_path (str): Path to the delay data CSV file.
        """
        self.df = pd.read_csv(delay_data_path)
        self._clean_data()

    def _clean_data(self):
        """
        Prepares and cleans delay dataset for plotting.
        """
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)

    def render_boxplot(self, delay_type="arrival", directions=None, top_n=10):
        """
        Render a Plotly boxplot for selected delay type and directions.

        Args:
            delay_type (str): 'arrival' or 'departure'
            directions (list, optional): List of directions to include. If None, auto-selects top N by frequency.
            top_n (int): If directions is None, use top N most frequent directions.
        """
        if delay_type not in ["arrival", "departure"]:
            raise ValueError("delay_type must be 'arrival' or 'departure'")

        col = "Delay at arrival" if delay_type == "arrival" else "Delay at departure"
        df = self.df[self.df[col] > 0].copy()

        # Select top N directions if none specified
        if directions is None:
            directions = df["Relation direction"].value_counts().nlargest(top_n).index.tolist()

        df = df[df["Relation direction"].isin(directions)]

        fig = px.box(
            df,
            x="Relation direction",
            y=col,
            points="outliers",
            color="Relation direction",
            labels={"Relation direction": "Direction", col: f"Delay ({delay_type}) (minutes)"},
            title=f"{delay_type.title()} Delay Distribution by Direction"
        )

        fig.update_layout(showlegend=False)
        return fig
