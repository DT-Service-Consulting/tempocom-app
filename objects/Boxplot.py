import pandas as pd
import plotly.graph_objects as go
import unicodedata
import streamlit as st

class BaseBoxPlotDB:
    TABLE_MAP = {
        "relation": "punctuality_boxplots",
        "station":  "punctuality_boxplots",
        "link":     "punctuality_boxplots",
    }

    def __init__(self, db_connector, plot_type: str):
        self.dbc = db_connector
        self.plot_type = plot_type
        self.table = self.TABLE_MAP[plot_type]
        self.df = self._load_data()
        self.df["_key"] = self.df["name"].map(_norm)

    def _load_data(self):
        sql = f"""
            SELECT planned,Q1, Q3, max, median, min, n_samples, name, outliers, type
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
            fig.update_layout(title=title, xaxis_title="Delay (minutes)")
            fig.add_annotation(
                text="No data for current selection", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False
            )
            return fig

        all_planned = []
        all_mins = []
        all_maxs = []

        for _, row in data.iterrows():
            # Boxplot values
            x_values = row["outliers"] + [row["Q1"], row["Q3"], row["median"], row["min"], row["max"]]
            all_mins.append(row["min"])
            all_maxs.append(row["max"])

            fig.add_trace(go.Box(
                x=x_values,
                name=row["name"],
                boxpoints="outliers" if row["outliers"] else False,
                hovertext=f"n={row['n_samples']}",
                hoverinfo="text",
                orientation="h"
            ))

            # Add star for planned value
            if not pd.isna(row.get("planned")):
                all_planned.append(row["planned"])
                fig.add_trace(go.Scatter(
                    x=[row["planned"]],
                    y=[row["name"]],
                    mode="markers+text",
                    marker=dict(
                        symbol="star-diamond",
                        size=20,
                        color="red",
                        line=dict(color="black", width=2)
                    ),
                    text=["Planned"],
                    textposition="middle right",
                    name="Planned",
                    showlegend=False
                ))

        # Extend x-axis to fit planned markers
        if all_planned:
            min_x = min(all_mins + all_planned)
            max_x = max(all_maxs + all_planned)
            fig.update_xaxes(range=[min_x * 0.95, max_x * 1.05])

        fig.update_layout(
            title=title,
            xaxis_title="Delay (minutes)",
            boxmode="group"
        )

        return fig


class DelayBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="relation")

    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Relation", data=data)




def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(str(s).split()).strip().casefold()




class DelayBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="relation")

    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Relation", data=data)


class StationBoxPlot(BaseBoxPlotDB):

    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='station')

    def get_stations_for_directions(self, directions):
        if not directions:
            return []

        direction_list = ",".join(f"'{d}'" for d in directions)

        # ✅ Step 1: Get station names from direction_stops
        sql = f"""
            SELECT DISTINCT station_name
            FROM direction_stops
            WHERE direction_name IN ({direction_list})
              AND station_name IS NOT NULL
        """
        direction_df = pd.read_sql(sql, self.dbc.conn)

        # ✅ Step 2: Normalize both sets for safe comparison
        direction_stations = set(direction_df["station_name"].dropna().str.strip().str.title())

        # ✅ Step 3: Filter self.df by those station names
        boxplot_stations = self.df["name"].dropna().str.strip().str.title()
        matched_stations = [s for s in boxplot_stations if s in direction_stations]

        return sorted(set(matched_stations))



    def render_boxplot(self, selected_directions):
        if selected_directions:
            stations = self.get_stations_for_directions(selected_directions)
            self.df = self.df[self.df['name'].str.strip().str.title().isin(stations)]
        return self._create_boxplot("Total Delay Distribution by Station")

class LinkBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='link')

    def get_links_for_directions(self, directions):
        name_list = ",".join(f"'{d}'" for d in directions)
        sql = f"""
        SELECT direction_name, station_name, order_in_route
        FROM direction_stops
        WHERE direction_name IN ({name_list})
        ORDER BY direction_name, order_in_route
        """
        df = pd.read_sql(sql, self.dbc.conn)
        df['station_name'] = df['station_name'].str.title()

        links = []
        for direction, group in df.groupby("direction_name"):
            stations = group.sort_values("order_in_route")["station_name"].tolist()
            for i in range(len(stations) - 1):
                links.append(f"{stations[i]} ⇔ {stations[i+1]}")
        return list(set(links))

    def render_boxplot(self, selected_directions=None):
        if selected_directions:
            links = self.get_links_for_directions(selected_directions)
            self.df = self.df[self.df['name'].isin(links)]
        return self._create_boxplot("Total Delay Distribution by Link")
