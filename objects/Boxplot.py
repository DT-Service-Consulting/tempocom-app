"""
Boxplot Visualization Module for Train Delays

This module provides classes for creating interactive boxplots of total train delays (arrival + departure) using Plotly.

Classes
-------
- DelayBoxPlot:
    Generates boxplots of total delay by train direction ("Relation direction").
- StationBoxPlot:
    Generates boxplots of total delay by station ("Stopping place (FR)").

Each class loads and cleans the delay data, computes total delay, and provides a method to render a boxplot for the top N directions or stations, or for a user-specified subset.

Dependencies
------------
- pandas
- plotly.express

Typical Usage
-------------
Instantiate the desired class with the path to the delay data CSV, then call the `render_boxplot` method to generate the plot.

Example:
    boxplot = DelayBoxPlot("delays.csv")
    fig = boxplot.render_boxplot()
    fig.show()
"""

import pandas as pd
import plotly.express as px


def get_direction_key(relation: str) -> str:
    cities = [c.strip() for c in relation.split("-")]
    return " - ".join(sorted(cities))


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
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["direction_key"] = self.df["Relation direction"].apply(get_direction_key)

    def get_bidirectional(self, direction: str):
        """
        Given a direction, return all actual relation names that are bidirectional.
        """
        key = get_direction_key(direction)
        return self.df[self.df["direction_key"] == key]["Relation direction"].unique().tolist()

    def get_relation_from_direction(self, direction: str):
        """
        Given a Relation direction, return the corresponding Relation (e.g., IC 11).
        """
        match = self.df[self.df["Relation direction"] == direction]["Relation"].dropna().unique()
        return match[0] if len(match) > 0 else None

    def get_directions_by_relation(self, relation: str):
        """
        Return all Relation direction values that belong to the given Relation (e.g., IC 11).
        """
        return self.df[self.df["Relation"] == relation]["Relation direction"].dropna().unique().tolist()

    def render_boxplot(self, directions=None):
        """
        Render a Plotly boxplot for total delay by direction.

        Args:
            directions (list, optional): List of directions to include. If None, auto-selects all available.
        """
        df = self.df[self.df["Total Delay"] > 0].copy()

        if directions:
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
            yaxis=dict(range=[0, 1000])
        )
        return fig

    def render_station_distribution_for_direction(self, direction: str):
        """
        Render a boxplot of total delay distribution per station for a specific direction.

        Args:
            direction (str): The relation direction to filter by.
        """
        df = self.df[
            (self.df["Total Delay"] > 0) &
            (self.df["Relation direction"] == direction)
        ].copy()

        if df.empty:
            return None

        fig = px.box(
            df,
            x="Stopping place (FR)",
            y="Total Delay",
            points="outliers",
            color="Stopping place (FR)",
            labels={
                "Stopping place (FR)": "Station",
                "Total Delay": "Delay (minutes)"
            },
            title=f"Delay Distribution per Station for {direction}"
        )

        fig.update_layout(
            showlegend=False,
            yaxis=dict(range=[0, 1000])
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
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]

    def render_boxplot(self, stations=None):
        """
        Render a Plotly boxplot for total delay by station.

        Args:
            stations (list, optional): List of stations to include. If None, auto-selects all available.
        """
        df = self.df[self.df["Total Delay"] > 0].copy()

        if stations:
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
            yaxis=dict(range=[0, 1000])
        )
        return fig



import pandas as pd
import plotly.express as px


class LinkBoxPlot:
    """
    Creates boxplots of total delays between consecutive stations in a selected direction.
    """

    def __init__(self, delay_data_path: str):
        """
        Initialize LinkBoxPlot with delay data CSV.

        Args:
            delay_data_path (str): Path to the delay data CSV file.
        """
        self.df = pd.read_csv(delay_data_path)
        self._clean_data()
        self.link_df = None  # Cached after computing links

    def _clean_data(self):
        """
        Clean and prepare the data.
        """
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]

    def _compute_links(self, direction: str):
        """
        Computes delay between consecutive stations in the selected direction.

        Args:
            direction (str): The relation direction to analyze.
        """
        df_dir = self.df[self.df["Relation direction"] == direction].copy()
        if df_dir.empty:
            return pd.DataFrame()

        # Sort by train and planned departure
        df_dir.sort_values(by=["Train number", "Planned departure time"], inplace=True)

        # For each train, compute station pairs and link-level delay
        records = []
        for train_id, group in df_dir.groupby("Train number"):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                from_stop = group.loc[i, "Stopping place (FR)"]
                to_stop = group.loc[i + 1, "Stopping place (FR)"]

                link_label = f"{from_stop} ➝ {to_stop}"

                delay = group.loc[i + 1, "Total Delay"]
                if delay > 0:
                    records.append({
                        "Train number": train_id,
                        "From": from_stop,
                        "To": to_stop,
                        "Link": link_label,
                        "Total Delay": delay
                    })

        return pd.DataFrame(records)

    def render_boxplot(self, direction: str):
        """
        Render boxplot of link-level total delay per consecutive station pair.

        Args:
            direction (str): The relation direction to analyze.
        """
        link_df = self._compute_links(direction)

        if link_df.empty:
            return None

        fig = px.box(
            link_df,
            x="Link",
            y="Total Delay",
            points="outliers",
            color="Link",
            labels={"Link": "Station Pair", "Total Delay": "Delay (minutes)"},
            title=f"Total Delay Between Consecutive Stations in {direction}"
        )

        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1000]))
        return fig
