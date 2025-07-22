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
import plotly.graph_objects as go


def get_direction_key(relation: str) -> str:
    cities = [c.strip() for c in relation.split("-")]
    return " - ".join(sorted(cities))


class DelayBoxPlot:
    """
    Creates boxplots of total train delays (arrival + departure) by direction,
    and overlays planning duration curves or bands.
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
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["direction_key"] = self.df["Relation direction"].apply(get_direction_key)
        self.df["Planned departure time"] = pd.to_datetime(self.df["Planned departure time"], errors="coerce", infer_datetime_format=True)
        self.df["Planned arrival time"] = pd.to_datetime(self.df["Planned arrival time"], errors="coerce", infer_datetime_format=True)

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

    def compute_planning_band(self, duration_df, threshold: float = 5.0):
        """
        Aggregates per relation to compute planning bands and identifies significant variance.

        Returns a DataFrame with mean, min, max, std, and variance flag.
        """
        summary = duration_df.groupby("Relation")["Planned Duration (min)"].agg(["mean", "min", "max", "std"]).reset_index()
        summary["Significant Variance"] = summary["std"] > threshold
        return summary

    def compute_planning_durations(self):
        """
        Vectorized computation of planned durations per train.
        """
        df = self.df.dropna(subset=["Planned departure time", "Planned arrival time"]).copy()

        # Ensure sorting by planned times
        df = df.sort_values(["Train number", "Planned departure time", "Planned arrival time"])

        # Get first departure time and relation per train
        dep_df = df.groupby("Train number").first().reset_index()[["Train number", "Planned departure time", "Relation"]]
        # Get last arrival time per train
        arr_df = df.groupby("Train number").last().reset_index()[["Train number", "Planned arrival time"]]

        # Merge
        merged = pd.merge(dep_df, arr_df, on="Train number")
        merged["Planned Duration (min)"] = (merged["Planned arrival time"] - merged["Planned departure time"]).dt.total_seconds() / 60.0

        return merged[["Train number", "Relation", "Planned Duration (min)"]]


    def render_boxplot(self, directions=None):
        """
        Render a Plotly boxplot for total delay by direction with planning overlay.

        Args:
            directions (list, optional): List of directions to include. If None, includes all.
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

        # Overlay planning durations
        durations_df = self.compute_planning_durations()
        band_df = self.compute_planning_band(durations_df)

        for direction in df["Relation direction"].unique():
            relation = self.get_relation_from_direction(direction)
            if relation and relation in band_df["Relation"].values:
                stats = band_df[band_df["Relation"] == relation].iloc[0]

                if stats["Significant Variance"]:
                    fig.add_hline(y=stats["min"], line_dash="dot", line_color="green",
                                  annotation_text=f"{relation} Min Plan", annotation_position="top left")
                    fig.add_hline(y=stats["mean"], line_dash="dash", line_color="blue",
                                  annotation_text=f"{relation} Mean Plan", annotation_position="top left")
                    fig.add_hline(y=stats["max"], line_dash="dot", line_color="red",
                                  annotation_text=f"{relation} Max Plan", annotation_position="top left")
                else:
                    fig.add_hline(y=stats["mean"], line_dash="dash", line_color="blue",
                                  annotation_text=f"{relation} Plan", annotation_position="top left")

        # Adjust Y-axis range dynamically based on highest planned duration
        max_planned = band_df["max"].max() if not band_df.empty else 1000
        max_observed = df["Total Delay"].max()
        y_max = max(max_planned, max_observed, 1000) * 1.1  # Add 10% buffer

        fig.update_layout(showlegend=False, yaxis=dict(range=[0, y_max]))
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
    Creates boxplots of total train delays (arrival + departure) by station,
    with planning duration overlay.
    """

    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._clean_data()

    def _clean_data(self):
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["Planned departure time"] = pd.to_datetime(self.df["Planned departure time"], errors="coerce", infer_datetime_format=True)
        self.df["Planned arrival time"] = pd.to_datetime(self.df["Planned arrival time"], errors="coerce", infer_datetime_format=True)
        self.df["Relation"] = self.df["Relation"].astype(str)
        self.df["Train number"] = self.df["Train number"].astype(str)

    def _compute_planning_durations(self):
        df = self.df.dropna(subset=["Planned departure time", "Planned arrival time"]).copy()
        df = df.sort_values(["Train number", "Planned departure time", "Planned arrival time"])

        dep_df = df.groupby("Train number").first().reset_index()[["Train number", "Planned departure time", "Relation"]]
        arr_df = df.groupby("Train number").last().reset_index()[["Train number", "Planned arrival time"]]
        merged = pd.merge(dep_df, arr_df, on="Train number")
        merged["Planned Duration (min)"] = (merged["Planned arrival time"] - merged["Planned departure time"]).dt.total_seconds() / 60.0

        return merged

    def _compute_planning_band(self, duration_df, threshold=5.0):
        summary = duration_df.groupby("Relation")["Planned Duration (min)"].agg(["mean", "min", "max", "std"]).reset_index()
        summary["Significant Variance"] = summary["std"] > threshold
        return summary

    def render_boxplot(self, stations=None):
        df = self.df[self.df["Total Delay"] > 0].copy()

        if stations:
            df = df[df["Stopping place (FR)"].isin(stations)]

        fig = px.box(
            df,
            x="Stopping place (FR)",
            y="Total Delay",
            points="outliers",
            color="Stopping place (FR)",
            labels={"Stopping place (FR)": "Station", "Total Delay": "Total Delay (minutes)"},
            title="Total Delay Distribution by Station"
        )

        durations_df = self._compute_planning_durations()
        band_df = self._compute_planning_band(durations_df)

        max_y = max(df["Total Delay"].max(), band_df["max"].max() if not band_df.empty else 0) * 1.1

        for rel in df["Relation"].unique():
            if rel in band_df["Relation"].values:
                stats = band_df[band_df["Relation"] == rel].iloc[0]
                x_vals = df["Stopping place (FR)"].unique().tolist()

                if stats["Significant Variance"]:
                    fig.add_trace(go.Scatter(x=x_vals, y=[stats["min"]] * len(x_vals),
                                             mode="lines", line=dict(color="green", dash="dot"),
                                             name="Planned Min Duration"))
                    fig.add_trace(go.Scatter(x=x_vals, y=[stats["mean"]] * len(x_vals),
                                             mode="lines", line=dict(color="blue", dash="dash"),
                                             name="Planned Mean Duration"))
                    fig.add_trace(go.Scatter(x=x_vals, y=[stats["max"]] * len(x_vals),
                                             mode="lines", line=dict(color="red", dash="dot"),
                                             name="Planned Max Duration"))
                else:
                    fig.add_trace(go.Scatter(x=x_vals, y=[stats["mean"]] * len(x_vals),
                                             mode="lines", line=dict(color="blue", dash="dash"),
                                             name="Planned Duration"))

        fig.update_layout(
            yaxis=dict(range=[0, max_y]),
            legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

        df_dir.sort_values(by=["Train number", "Planned departure time"], inplace=True)

        records = []

        for train_id, group in df_dir.groupby("Train number"):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                from_stop = group.loc[i, "Stopping place (FR)"]
                to_stop = group.loc[i + 1, "Stopping place (FR)"]

                # 🚫 Skip links where start and end station are the same
                if from_stop == to_stop:
                    continue

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
