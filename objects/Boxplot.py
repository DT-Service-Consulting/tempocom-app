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
        "relation": "punctuality_boxplots",
        "station":  "punctuality_boxplots_station",
        "link":     "punctuality_boxplots_link",
    }

    def __init__(self, db_connector, plot_type: str):
        self.dbc = db_connector
        self.plot_type = plot_type
        self.table = self.TABLE_MAP[plot_type]
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
            y_values = row["outliers"] + [row["Q1"], row["Q3"], row["median"], row["min"], row["max"]]
            fig.add_trace(go.Box(
                y=y_values,
                name=row["name"],
                boxpoints="outliers" if row["outliers"] else False,
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
        super().__init__(db_connector, plot_type='station')
    


    def get_direction_ids(self, direction_names):
        name_list = ",".join(f"'{name}'" for name in direction_names)
        sql = f"""
        SELECT DISTINCT direction_id
        FROM direction_stops
        WHERE direction IN ({name_list})
        """
        df = pd.read_sql(sql, self.dbc.conn)
        return df['direction_id'].tolist()

    def get_stations_for_directions(self, direction_names):
        direction_ids = self.get_direction_ids(direction_names)
        if not direction_ids:
            return []

        id_list = ",".join(str(d) for d in direction_ids)
        sql = f"""
        SELECT DISTINCT ds.station_id, op.Complete_name_in_French AS station_name
        FROM direction_stops ds
        JOIN operational_points op ON ds.station_id = op.PTCAR_ID
        WHERE ds.direction_id IN ({id_list})
        """
        df = pd.read_sql(sql, self.dbc.conn)
        return df['station_name'].dropna().str.title().unique().tolist()



    def render_boxplot(self, selected_directions=None):
        if selected_directions:
            stations = self.get_stations_for_directions(selected_directions)
            self.df = self.df[self.df['name'].isin(stations)]
        return self._create_boxplot("Total Delay Distribution by Station")


class LinkBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='link')

    def get_links_for_directions(self, directions):
        direction_list = ",".join(f"'{d}'" for d in directions)
        
        sql = f"""
        SELECT ds.direction, ds.PTCAR_ID, ds.position, op.Complete_name_in_French AS station_name
        FROM direction_stops ds
        JOIN operational_points op ON ds.PTCAR_ID = op.PTCAR_ID
        WHERE ds.direction IN ({direction_list})
        ORDER BY ds.direction, ds.position
        """
        
        df = pd.read_sql(sql, self.dbc.conn)
        df['station_name'] = df['station_name'].str.title()
        
        links = []
        for dir_name, group in df.groupby('direction'):
            stations = group.sort_values('position')['station_name'].tolist()
            for i in range(len(stations) - 1):
                links.append(f"{stations[i]} ⇔ {stations[i+1]}")
        
        return list(set(links))


    def render_boxplot(self, selected_directions=None):
        if selected_directions:
            links = self.get_links_for_directions(selected_directions)
            self.df = self.df[self.df['name'].isin(links)]
        return self._create_boxplot("Total Delay Distribution by Link")
