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
            fig.update_layout(title=title, yaxis_title="Delay (minutes)")
            fig.add_annotation(
                text="No data for current selection", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False
            )
            return fig

        all_planned = []
        all_mins = []
        all_maxs = []

        for _, row in data.iterrows():
            y_values = row["outliers"] + [row["Q1"], row["Q3"], row["median"], row["min"], row["max"]]
            all_mins.append(row["min"])
            all_maxs.append(row["max"])

            fig.add_trace(go.Box(
                y=y_values,
                name=row["name"],
                boxpoints="outliers" if row["outliers"] else False,
                hovertext=f"n={row['n_samples']}",
                hoverinfo="text",
                orientation="v"
            ))
            if all_planned:
                    fig.add_trace(go.Scatter(
                        x=[row["name"] for _, row in data.iterrows() if not pd.isna(row.get("planned"))],
                        y=[row["planned"] for _, row in data.iterrows() if not pd.isna(row.get("planned"))],
                        mode="markers+text",
                        marker=dict(
                            symbol="star-diamond",
                            size=[max(12, min(30, (row["max"] - row["min"]) // 10)) for _, row in data.iterrows() if not pd.isna(row.get("planned"))],
                            color="red",
                            line=dict(color="black", width=2)
                        ),
                        text=["Planned"] * len([_ for _, row in data.iterrows() if not pd.isna(row.get("planned"))]),
                        textposition="top right",
                        name="Planned",
                        showlegend=False
                    ))


            if not pd.isna(row.get("planned")):
                all_planned.append(row["planned"])
                fig.add_trace(go.Scatter(
                    x=[row["name"]],
                    y=[row["planned"]],
                    mode="markers+text",
                    marker=dict(
                        symbol="star-diamond",
                        size=max(12, min(30, (row["max"] - row["min"]) // 10)),
                        color="red",
                        line=dict(color="black", width=2)
                    ),
                    text=["Planned"],
                    textposition="top right",
                    name="Planned",
                    showlegend=False
                ))

        if all_planned:
            min_y = min(all_mins + all_planned)
            max_y = max(all_maxs + all_planned)
            fig.update_yaxes(range=[min_y * 0.95, max_y * 1.05])

        fig.update_layout(
            title=title,
            yaxis_title="Delay (minutes)",
            boxmode="group"
        )

        return fig



class DelayBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type="relation")

    def _create_boxplot(self, title, data=None):
        data = self.df if data is None else data
        fig = go.Figure()

        if data.empty:
            fig.update_layout(title=title, yaxis_title="Delay (minutes)")
            fig.add_annotation(
                text="No data for current selection", x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False
            )
            return fig

        all_mins = []
        all_maxs = []
        all_planned = []
        categories = data["name"].tolist()

        for _, row in data.iterrows():
            y_values = row["outliers"] + [row["Q1"], row["Q3"], row["median"], row["min"], row["max"]]
            all_mins.append(row["min"])
            all_maxs.append(row["max"])

            fig.add_trace(go.Box(
                y=y_values,
                name=row["name"],
                boxpoints="outliers" if row["outliers"] else False,
                hovertext=f"n={row['n_samples']}",
                hoverinfo="text",
                orientation="v"
            ))

        # Add dashed line with transparent marker for hover
        for idx, row in enumerate(data.itertuples()):
            if not pd.isna(row.planned):
                all_planned.append(row.planned)
                fig.add_shape(
                    type="line",
                    x0=idx - 0.4, x1=idx + 0.4,
                    y0=row.planned, y1=row.planned,
                    xref="x", yref="y",
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_trace(go.Scatter(
                    x=[idx],
                    y=[row.planned],
                    mode="markers",
                    marker=dict(size=8, color="rgba(0,0,0,0)"),
                    hovertemplate=f"<b>{row.name}</b><br>Planned: {row.planned} min<extra></extra>",
                    showlegend=False
                ))

        if all_planned:
            min_y = min(all_mins + all_planned)
            max_y = max(all_maxs + all_planned)
            fig.update_yaxes(range=[min_y * 0.95, max_y * 1.05])
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Planned',
                hoverinfo="skip"
            ))

        fig.update_layout(
            title=title,
            yaxis_title="Delay (minutes)",
            boxmode="group",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        )

        return fig

    def render_boxplot(self, filter_names=None):
        data = self._filter_df(filter_names)
        return self._create_boxplot("Total Delay Distribution by Relation", data=data)

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(str(s).split()).strip().casefold()





class StationBoxPlot(BaseBoxPlotDB):

    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='station')

    def get_stations_for_directions(conn, selected_directions):
        placeholders = ','.join(['?'] * len(selected_directions))
        query = f"""
            SELECT DISTINCT name AS station_name
            FROM punctuality_boxplots
            WHERE type = 'station' AND name IN ({placeholders})
        """
        df = pd.read_sql(query, conn, params=selected_directions)
        return df["station_name"].dropna().str.strip().str.title().tolist()
        



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
