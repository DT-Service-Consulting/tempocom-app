import pandas as pd
import plotly.express as px

class DelayBoxPlot:
    """
    Creates boxplots of total train delays (arrival + departure) by direction.
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
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)

        # Compute total delay
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]

    def render_boxplot(self, directions=None, top_n=10):
        """
        Render a Plotly boxplot for total delay by direction.

        Args:
            directions (list, optional): List of directions to include. If None, auto-selects top N by frequency.
            top_n (int): If directions is None, use top N most frequent directions.
        """
        df = self.df[self.df["Total Delay"] > 0].copy()

        # Select top N directions if none specified
        if directions is None:
            directions = df["Relation direction"].value_counts().nlargest(top_n).index.tolist()

        df = df[df["Relation direction"].isin(directions)]

        fig = px.box(
            df,
            x="Relation direction",
            y="Total Delay",
            points="outliers",
            color="Relation direction",
            labels={
                "Relation direction": "Direction",
                "Total Delay": "Total Delay (minutes)"
            },
            title="Total Delay Distribution by Direction"
        )

        fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 1000])  # Set visible y-axis range
                )
        return fig





class StationBoxPlot:
    """
    Creates boxplots of total train delays (arrival + departure) by station.
    """

    def __init__(self, delay_data_path: str):
        """
        Initialize StationBoxPlot with delay data CSV.

        Args:
            delay_data_path (str): Path to the delay data CSV file.
        """
        self.df = pd.read_csv(delay_data_path)
        self._clean_data()

    def _clean_data(self):
        """
        Prepares and cleans delay dataset for plotting.
        """
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()

        # Compute total delay
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]

    def render_boxplot(self, stations=None, top_n=10):
        """
        Render a Plotly boxplot for total delay by station.

        Args:
            stations (list, optional): List of stations to include. If None, auto-selects top N by frequency.
            top_n (int): If stations is None, use top N most frequent stations.
        """
        df = self.df[self.df["Total Delay"] > 0].copy()

        # Select top N stations if none specified
        if stations is None:
            stations = df["Stopping place (FR)"].value_counts().nlargest(top_n).index.tolist()

        df = df[df["Stopping place (FR)"].isin(stations)]

        fig = px.box(
            df,
            x="Stopping place (FR)",
            y="Total Delay",
            points="outliers",
            color="Stopping place (FR)",
            labels={
                "Stopping place (FR)": "Station",
                "Total Delay": "Total Delay (minutes)"
            },
            title="Total Delay Distribution by Station"
        )

        fig.update_layout(
            showlegend=False,
            yaxis=dict(range=[0, 1000])  # Limit visual range on Y-axis
        )
        return fig