
import pandas as pd
import plotly.graph_objects as go

class BaseBoxPlotDB:
    def __init__(self, db_connector, plot_type):
        self.dbc = db_connector
        self.plot_type = plot_type
        self.df = self._load_data()

    def _load_data(self):
        sql = f"""
            SELECT Q1, Q3, max, median, min, n_samples, name, outliers, type
            FROM punctuality_boxplots
            WHERE type = '{self.plot_type}'
        """
        df = pd.read_sql(sql, self.dbc.conn)

        df['outliers'] = df['outliers'].fillna('').apply(lambda x: [int(v) for v in str(x).split(',') if v.strip().isdigit()])
        return df

    def _create_boxplot(self, title):
        fig = go.Figure()
        for _, row in self.df.iterrows():
            fig.add_trace(go.Box(
                q1=[row['Q1']],
                q3=[row['Q3']],
                median=[row['median']],
                lowerfence=[row['min']],
                upperfence=[row['max']],
                boxpoints='outliers',
                y=row['outliers'] if row['outliers'] else None,
                name=row['name']
            ))
        fig.update_layout(title=title, yaxis_title="Delay (seconds)")
        return fig


class DelayBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='relation')

    def render_boxplot(self, filter_names=None):
        if filter_names:
            self.df = self.df[self.df['name'].isin(filter_names)]
        return self._create_boxplot("Total Delay Distribution by Relation")


class StationBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='station')

    def render_boxplot(self, filter_names=None):
        if filter_names:
            self.df = self.df[self.df['name'].isin(filter_names)]
        return self._create_boxplot("Total Delay Distribution by Station")


class LinkBoxPlot(BaseBoxPlotDB):
    def __init__(self, db_connector):
        super().__init__(db_connector, plot_type='link')

    def render_boxplot(self, filter_names=None):
        if filter_names:
            self.df = self.df[self.df['name'].isin(filter_names)]
        return self._create_boxplot("Total Delay Distribution by Link")
