"""
Boxplot Visualization Module for Train Delays

This module provides classes for creating interactive boxplots of total train delays (arrival + departure) using Plotly.

Classes
-------
- DelayBoxPlot:
    Generates boxplots of total delay by train direction ("Relation direction").
- StationBoxPlot:
    Generates boxplots of total delay by station ("Stopping place (FR)").
- LinkBoxPlot:
    Generates boxplots of delay between consecutive stations.

Each class uses a shared DataFrame to avoid repeated loading from disk and defers heavy processing until rendering time.

Dependencies
------------
- pandas
- plotly.express
- plotly.graph_objects
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_direction_key(relation: str) -> str:
    cities = [c.strip() for c in relation.split("-")]
    return " - ".join(sorted(cities))


def compute_planning_durations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Planned departure time", "Planned arrival time"]).copy()
    df = df.sort_values(["Train number", "Planned departure time", "Planned arrival time"])

    dep_df = df.groupby("Train number").first().reset_index()[["Train number", "Planned departure time", "Relation"]]
    arr_df = df.groupby("Train number").last().reset_index()[["Train number", "Planned arrival time"]]
    merged = pd.merge(dep_df, arr_df, on="Train number")
    merged["Planned Duration (min)"] = (merged["Planned arrival time"] - merged["Planned departure time"]).dt.total_seconds() / 60.0
    return merged


def compute_planning_band(duration_df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    summary = duration_df.groupby("Relation")["Planned Duration (min)"].agg(["mean", "min", "max", "std"]).reset_index()
    summary["Significant Variance"] = summary["std"] > threshold
    return summary


class DelayBoxPlot:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._cleaned = False
        self._planning_durations = None
        self._band_df = None

    def _clean_data(self):
        if self._cleaned:
            return
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["direction_key"] = self.df["Relation direction"].apply(get_direction_key)
        self.df["Planned departure time"] = pd.to_datetime(self.df["Planned departure time"], errors="coerce")
        self.df["Planned arrival time"] = pd.to_datetime(self.df["Planned arrival time"], errors="coerce")
        self._cleaned = True

    def _ensure_planning_data(self):
        if self._planning_durations is None or self._band_df is None:
            self._planning_durations = compute_planning_durations(self.df)
            self._band_df = compute_planning_band(self._planning_durations)

    def get_relation_from_direction(self, direction: str):
        result = self.df.loc[self.df["Relation direction"] == direction, "Relation"].dropna().unique()
        return result[0] if len(result) else None

    def get_directions_by_relation(self, relation: str):
        return self.df[self.df["Relation"] == relation]["Relation direction"].dropna().unique().tolist()

    def render_boxplot(self, directions=None):
        self._clean_data()
        df = self.df.query("`Total Delay` > 0")
        if directions:
            df = df[df["Relation direction"].isin(directions)]
        if df.empty:
            return None

        self._ensure_planning_data()
        fig = px.box(df, x="Relation direction", y="Total Delay", points="outliers", color="Relation direction",
                     labels={"Relation direction": "Direction", "Total Delay": "Delay (min)"},
                     title="Total Delay Distribution by Direction")

        for direction in df["Relation direction"].unique():
            rel = self.get_relation_from_direction(direction)
            if rel and rel in self._band_df["Relation"].values:
                stats = self._band_df[self._band_df["Relation"] == rel].iloc[0]
                if stats["Significant Variance"]:
                    fig.add_hline(y=stats["min"], line_dash="dot", line_color="green",
                                  annotation_text=f"{rel} Min", annotation_position="top left")
                    fig.add_hline(y=stats["mean"], line_dash="dash", line_color="blue",
                                  annotation_text=f"{rel} Mean", annotation_position="top left")
                    fig.add_hline(y=stats["max"], line_dash="dot", line_color="red",
                                  annotation_text=f"{rel} Max", annotation_position="top left")
                else:
                    fig.add_hline(y=stats["mean"], line_dash="dash", line_color="blue",
                                  annotation_text=f"{rel} Plan", annotation_position="top left")

        ymax = max(df["Total Delay"].max(), self._band_df["max"].max() if not self._band_df.empty else 0) * 1.1
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, ymax]))
        return fig

    def render_station_distribution_for_direction(self, direction: str):
        self._clean_data()
        df = self.df.query("`Total Delay` > 0 and `Relation direction` == @direction")
        if df.empty:
            return None

        fig = px.box(df, x="Stopping place (FR)", y="Total Delay", points="outliers", color="Stopping place (FR)",
                     labels={"Stopping place (FR)": "Station", "Total Delay": "Delay (min)"},
                     title=f"Delay Distribution per Station for {direction}")
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1000]))
        return fig


class StationBoxPlot:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._cleaned = False
        self._planning_durations = None
        self._band_df = None

    def _clean_data(self):
        if self._cleaned:
            return
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Relation"] = self.df["Relation"].astype(str)
        self.df["Planned departure time"] = pd.to_datetime(self.df["Planned departure time"], errors="coerce")
        self.df["Planned arrival time"] = pd.to_datetime(self.df["Planned arrival time"], errors="coerce")
        self._cleaned = True

    def _ensure_planning_data(self):
        if self._planning_durations is None or self._band_df is None:
            self._planning_durations = compute_planning_durations(self.df)
            self._band_df = compute_planning_band(self._planning_durations)

    def render_boxplot(self, stations=None):
        self._clean_data()
        df = self.df.query("`Total Delay` > 0")
        if stations:
            df = df[df["Stopping place (FR)"].isin(stations)]
        if df.empty:
            return None

        self._ensure_planning_data()
        fig = px.box(df, x="Stopping place (FR)", y="Total Delay", points="outliers", color="Stopping place (FR)",
                     labels={"Stopping place (FR)": "Station", "Total Delay": "Delay (min)"},
                     title="Total Delay Distribution by Station")

        ymax = max(df["Total Delay"].max(), self._band_df["max"].max() if not self._band_df.empty else 0) * 1.1

        for rel in df["Relation"].unique():
            if rel in self._band_df["Relation"].values:
                stats = self._band_df[self._band_df["Relation"] == rel].iloc[0]
                x_vals = df["Stopping place (FR)"].unique().tolist()
                lines = []
                if stats["Significant Variance"]:
                    lines = [
                        (stats["min"], "green", "dot", "Min"),
                        (stats["mean"], "blue", "dash", "Mean"),
                        (stats["max"], "red", "dot", "Max"),
                    ]
                else:
                    lines = [(stats["mean"], "blue", "dash", "Plan")]

                for val, color, dash, name in lines:
                    fig.add_trace(go.Scatter(x=x_vals, y=[val] * len(x_vals),
                                             mode="lines", line=dict(color=color, dash=dash), name=f"{rel} {name}"))

        fig.update_layout(yaxis=dict(range=[0, ymax]),
                          legend=dict(title="Legend", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig


class LinkBoxPlot:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._cleaned = False

    def _clean_data(self):
        if self._cleaned:
            return
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce").fillna(0)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce").fillna(0)
        self.df["Total Delay"] = self.df["Delay at arrival"] + self.df["Delay at departure"]
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Planned departure time"] = pd.to_datetime(self.df["Planned departure time"], errors="coerce")
        self._cleaned = True

    def _compute_links(self, direction: str):
        self._clean_data()
        df_dir = self.df[self.df["Relation direction"] == direction]
        if df_dir.empty:
            return pd.DataFrame()

        df_dir = df_dir.sort_values(by=["Train number", "Planned departure time"])
        records = []

        for train_id, group in df_dir.groupby("Train number"):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                from_stop = group.loc[i, "Stopping place (FR)"]
                to_stop = group.loc[i + 1, "Stopping place (FR)"]
                if from_stop == to_stop:
                    continue
                delay = group.loc[i + 1, "Total Delay"]
                if delay > 0:
                    records.append({
                        "Train number": train_id,
                        "From": from_stop,
                        "To": to_stop,
                        "Link": f"{from_stop} ➝ {to_stop}",
                        "Total Delay": delay
                    })

        return pd.DataFrame(records)

    def render_boxplot(self, direction: str):
        link_df = self._compute_links(direction)
        if link_df.empty:
            return None

        fig = px.box(link_df, x="Link", y="Total Delay", points="outliers", color="Link",
                     labels={"Link": "Station Pair", "Total Delay": "Delay (min)"},
                     title=f"Total Delay Between Consecutive Stations in {direction}")
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1000]))
        return fig