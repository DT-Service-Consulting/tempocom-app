# objects/Boxplot.py

import pandas as pd
import plotly.graph_objects as go
import unicodedata

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(str(s).split()).strip().casefold()

class BaseBoxPlotDB:
    TABLE_MAP = {
        "relation": "punctuality_boxplots",          # unchanged
        "station":  "punctuality_boxplots_station",  # ✅ new table
        "link":     "punctuality_boxplots_link",     # ✅ new table
    }

    def __init__(self, db_connector, plot_type: str):
        self.dbc = db_connector
        self.plot_type = plot_type
        self.table = self.TABLE_MAP[plot_type]       # pick the right table
        self.df = self._load_data()
        self.df["_key"] = self.df["name"].map(_norm)

    def _load_data(self):
        sql = f"""
            SELECT Q1, Q3, max, median, min, n_samples, name, outliers, type
            FROM {self.table}
            WHERE type = '{self.plot_type}'
        """
        df = pd.read_sql(sql, self.dbc.conn)
        df["outliers"] = (
            df["outliers"].fillna("")
            .apply(lambda x: [int(v) for v in str(x).split(",") if str(v).strip().isdigit()])
        )
        return df

    def _filter_df(self, filter_names):
        if not filter_names:
            return self.df
        keys = {_norm(n) for n in filter_names if str(n).strip()}
        return self.df[self.df["_key"].isin(keys)]

    def _create_boxplot(self, title, data=None):
        data = self.df if data is None else data
        fig = go.Figure()
        if data.empty:
            fig.update_layout(title=title, yaxis_title="Delay (seconds)")
            fig.add_annotation(text="No data for current selection", x=0.5, y=0.5, xref="paper", yref="paper",
                               showarrow=False)
            return fig

        for _, row in data.iterrows():
            has_outliers = bool(row["outliers"])
            fig.add_trace(go.Box(
                q1=[row["Q1"]],
                q3=[row["Q3"]],
                median=[row["median"]],
                lowerfence=[row["min"]],
                upperfence=[row["max"]],
                name=row["name"],
                boxpoints="outliers" if has_outliers else False,
                y=row["outliers"] if has_outliers else None,
                hovertext=f"n={row['n_samples']}",
                hoverinfo="text",
            ))
        fig.update_layout(title=title, yaxis_title="Delay (seconds)", boxmode="group")
        return fig

class DelayBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="relation")
    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Relation", data=data)

class StationBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="station")
    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Station", data=data)

class LinkBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="link")
    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Link", data=data)
