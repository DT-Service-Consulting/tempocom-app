"""
Delay_network.py

This module provides classes for visualizing train delay data using interactive bubble maps and heatmaps.

Classes:
- DelayBubbleMap: Visualizes total arrival delays per station as a Folium bubble map.
- DelayBubbleMap2: Visualizes total departure delays per station as a Folium bubble map.
- DelayHeatmap: Generates Plotly heatmaps of delays (arrival or departure) by station and hour.

Features:
- Reads station and delay data from CSV files.
- Aggregates and filters delay data for visualization.
- Supports filtering by station.
- Produces interactive maps and heatmaps for use in dashboards.

Dependencies:
- pandas
- folium
- ast
- matplotlib
- seaborn
- plotly

Author: Mohamad Hussain
Date: [2025-06-20]
"""# Updated Delay_network.py with support for date filtering and severity coloring

import pandas as pd
import plotly.express as px
import warnings
import folium
import ast
import plotly.graph_objects as go

import pandas as pd
import folium
import ast

import folium
import ast
import datetime
import datetime
import re
from typing import Iterable, Optional, Tuple, Union

import pandas as pd
import folium


DateLike = Union[str, datetime.date, datetime.datetime]
DateRange = Tuple[DateLike, DateLike]


import datetime
import re
from typing import Iterable, Optional, Tuple, Union

import datetime, re, math
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
import folium

DateLike = Union[str, datetime.date, datetime.datetime]
DateRange = Tuple[DateLike, DateLike]

class DelayBubbleMap:
    """
    One bubble per station (by default) with size ~ delay.
    Optionally, jitter overlapping points to see multiple records.
    """

    def __init__(self, conn,
                 min_radius: float = 4.0,
                 max_radius: float = 28.0,
                 aggregate: bool = True,   # <— DEFAULT: aggregate to one circle per station
                 agg_metric: str = "p90",  # mean|median|max|sum|p90
                 jitter_meters: float = 0.0):  # set to e.g. 30 to separate stacked points ~30m
        self.conn = conn
        self.merged = pd.DataFrame()
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.aggregate = bool(aggregate)
        self.agg_metric = agg_metric.lower()
        self.jitter_meters = float(jitter_meters)

    # ------------- helpers -------------
    @staticmethod
    def _to_ymd(d: DateLike) -> str:
        if isinstance(d, datetime.datetime): d = d.date()
        if isinstance(d, datetime.date): return d.strftime("%Y-%m-%d")
        s = str(d).strip().replace("/", "-")
        m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", s)  # dd-mm-yyyy
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else s

    @staticmethod
    def _parse_geo_point(val) -> Optional[tuple]:
        if val is None or (isinstance(val, float) and pd.isna(val)): return None
        s = str(val)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
        return (float(m.group(1)), float(m.group(2))) if m else None

    @staticmethod
    def _casefold_in(series: pd.Series, candidates: Iterable[str]) -> pd.Series:
        cand = {str(x).casefold().strip() for x in candidates}
        return series.astype("string").str.casefold().isin(cand)

    @staticmethod
    def _agg_func(name: str):
        name = name.lower()
        if name == "mean": return "mean"
        if name == "median": return "median"
        if name == "max": return "max"
        if name == "sum": return "sum"
        if name == "p90": return lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90))
        raise ValueError("agg_metric must be one of mean|median|max|sum|p90")

    @staticmethod
    def _jitter_point(pt: Tuple[float, float], meters: float, seed: Optional[int] = None) -> Tuple[float, float]:
        """Randomly offset a lat/lon by ~meters in both directions."""
        if meters <= 0: return pt
        rng = np.random.default_rng(seed)
        lat, lon = float(pt[0]), float(pt[1])
        # ~1 deg lat ≈ 111_111 m; 1 deg lon ≈ 111_111 * cos(lat)
        dlat = (rng.uniform(-1, 1) * meters) / 111_111.0
        denom = 111_111.0 * max(1e-6, math.cos(math.radians(lat)))
        dlon = (rng.uniform(-1, 1) * meters) / denom
        return lat + dlat, lon + dlon

    # ------------- core -------------
    def prepare_data(
        self,
        arrival: bool = True,
        station_filter: Optional[Union[str, Iterable[str]]] = None,
        date_filter: Optional[Union[DateLike, DateRange]] = None,
    ):
        delay_col = "DELAY_ARR" if arrival else "DELAY_DEP"
        time_col  = "REAL_DATE_ARR" if arrival else "REAL_DATE_DEP"

        # date handling
        if date_filter is None:
            start_date = end_date = datetime.date.today()
        elif isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
            start_date, end_date = date_filter
        else:
            start_date = end_date = date_filter
        ymd_start, ymd_end = self._to_ymd(start_date), self._to_ymd(end_date)

        # int-to-int join; pull Geo_Point + station name in one go
        sql = f"""
        SELECT
            CAST(p.{delay_col} AS FLOAT) / 60.0 AS Total_Delay_Minutes,
            p.{time_col} AS time,
            op.Complete_name_in_French AS station,
            op.Geo_Point AS Geo_Point
        FROM punctuality_public AS p
        INNER JOIN operational_points AS op
            ON p.STOPPING_PLACE_ID = op.PTCAR_ID
        WHERE p.{time_col} BETWEEN '{ymd_start}' AND '{ymd_end}'
        """
        raw = self.conn.query(sql)
        df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(
            raw, columns=["Total_Delay_Minutes", "time", "station", "Geo_Point"]
        )
        if df.empty:
            self.merged = pd.DataFrame()
            return

        # types + cleanup
        df["Total_Delay_Minutes"] = pd.to_numeric(df["Total_Delay_Minutes"], errors="coerce")
        df["station"] = df["station"].astype("string").str.strip().str.title()
        df["Geo_Point"] = df["Geo_Point"].apply(self._parse_geo_point)
        df = df.dropna(subset=["Total_Delay_Minutes", "Geo_Point", "station"])
        if df.empty:
            self.merged = pd.DataFrame()
            return

        # optional station filter
        if station_filter:
            if isinstance(station_filter, str): station_filter = [station_filter]
            df = df[self._casefold_in(df["station"], station_filter)]
            if df.empty:
                self.merged = pd.DataFrame()
                return

        # KEY FIX: collapse stacked points (same lat/lon) to ONE bubble per station
        if self.aggregate:
            func = self._agg_func(self.agg_metric)
            df = (df.groupby(["station", "Geo_Point"], as_index=False)
                    .agg(Total_Delay_Minutes=("Total_Delay_Minutes", func),
                         n=("Total_Delay_Minutes", "size")))
        else:
            # optional jitter to separate overlapping markers slightly
            if self.jitter_meters > 0:
                # use a deterministic seed per row so map is stable between reruns
                df = df.copy()
                df["Geo_Point"] = [self._jitter_point(pt, self.jitter_meters, seed=i)
                                   for i, pt in enumerate(df["Geo_Point"])]

        # sort so **smaller** circles draw first, **larger** last (visible on top)
        df = df.sort_values("Total_Delay_Minutes")
        self.merged = df[["station", "Geo_Point", "Total_Delay_Minutes"]].reset_index(drop=True)

    def render_map(self):
        if self.merged.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        df = self.merged.copy()
        delays = pd.to_numeric(df["Total_Delay_Minutes"], errors="coerce")
        df = df.loc[delays.notna()]
        if df.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        lats = [pt[0] for pt in df["Geo_Point"]]
        lons = [pt[1] for pt in df["Geo_Point"]]
        center = (float(np.mean(lats)), float(np.mean(lons)))
        m = folium.Map(location=center, zoom_start=7, tiles="cartodb positron")

        d = df["Total_Delay_Minutes"].to_numpy(dtype=float)
        dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
        if not math.isfinite(dmin) or not math.isfinite(dmax) or dmax <= dmin:
            dmax = dmin + 1e-6  # avoid collapse

        # strict linear mapping: delay -> radius
        sizes = np.interp(d, (dmin, dmax), (self.min_radius, self.max_radius)).astype(float)

        # simple green->red by absolute delay
        span = (dmax - dmin) if (dmax - dmin) != 0 else 1e-6
        def color_for(val: float) -> str:
            t = (val - dmin) / span
            r = int(255 * t); g = int(255 * (1 - t))
            return f"#{r:02x}{g:02x}00"

        # draw in ascending size so bigger bubbles end up on top
        for (station, (lat, lon), delay_min, radius) in zip(
            df["station"], df["Geo_Point"], d, sizes
        ):
            folium.CircleMarker(
                location=(float(lat), float(lon)),
                radius=float(radius),
                color=color_for(float(delay_min)),
                fill=True,
                fill_color=color_for(float(delay_min)),
                fill_opacity=0.7,
                popup=f"{station}<br>Delay: {round(float(delay_min), 1)} min"
            ).add_to(m)

        return m




import datetime, re, math, numpy as np, pandas as pd, folium
from typing import Optional, Union, Iterable, Tuple

DateLike   = Union[str, datetime.date, datetime.datetime]
DateRange  = Tuple[DateLike, DateLike]

class DelayBubbleMap2:
    """
    Delay bubble map (departure delays).
    - One bubble per station by default (aggregate=True, agg_metric='p90').
    - Optional jitter to separate overlapping points if you want every record.
    - Circle size strictly proportional to delay (linear min..max).
    """

    def __init__(self, conn,
                 min_radius: float = 4.0,
                 max_radius: float = 28.0,
                 aggregate: bool = True,
                 agg_metric: str = "p90",   # mean|median|max|sum|p90
                 jitter_meters: float = 0.0):
        self.conn = conn
        self.merged = pd.DataFrame()
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.aggregate = bool(aggregate)
        self.agg_metric = str(agg_metric).lower()
        self.jitter_meters = float(jitter_meters)

    # ---------- helpers ----------
    @staticmethod
    def _to_ymd(d: DateLike) -> str:
        """Return YYYY-MM-DD for SQL date comparisons."""
        if isinstance(d, datetime.datetime):
            d = d.date()
        if isinstance(d, datetime.date):
            return d.strftime("%Y-%m-%d")
        s = str(d).strip().replace("/", "-")
        # dd-mm-yyyy → yyyy-mm-dd
        m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", s)
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else s

    @staticmethod
    def _parse_geo_point(val) -> Optional[tuple]:
        """Parse 'lat, lon' string → (lat, lon) floats."""
        if val is None or (isinstance(val, float) and pd.isna(val)): return None
        s = str(val)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
        return (float(m.group(1)), float(m.group(2))) if m else None

    @staticmethod
    def _casefold_in(series: pd.Series, candidates: Iterable[str]) -> pd.Series:
        cand = {str(x).casefold().strip() for x in candidates}
        return series.astype("string").str.casefold().isin(cand)

    @staticmethod
    def _agg_func(name: str):
        name = name.lower()
        if name == "mean":   return "mean"
        if name == "median": return "median"
        if name == "max":    return "max"
        if name == "sum":    return "sum"
        if name == "p90":    return lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90))
        raise ValueError("agg_metric must be one of mean|median|max|sum|p90")

    @staticmethod
    def _jitter_point(pt: Tuple[float, float], meters: float, seed: Optional[int] = None) -> Tuple[float, float]:
        """Randomly offset a lat/lon by ~meters to separate overlapping markers."""
        if meters <= 0: return pt
        rng = np.random.default_rng(seed)
        lat, lon = float(pt[0]), float(pt[1])
        # 1 deg lat ≈ 111_111 m; 1 deg lon ≈ 111_111 * cos(lat)
        dlat = (rng.uniform(-1, 1) * meters) / 111_111.0
        denom = 111_111.0 * max(1e-6, math.cos(math.radians(lat)))
        dlon = (rng.uniform(-1, 1) * meters) / denom
        return lat + dlat, lon + dlon

    # ---------- core ----------
    def prepare_data(self, station_filter: Optional[Union[str, Iterable[str]]] = None,
                     date_filter: Optional[Union[DateLike, DateRange]] = None):
        """
        Build self.merged with columns: station, Geo_Point, Total_Delay_Minutes.
        - date_filter can be a single date or a (start, end) tuple.
        - Always uses DELAY_DEP and REAL_DATE_DEP.
        """
        # date handling
        if date_filter is None:
            start_date = end_date = datetime.date.today()
        elif isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
            start_date, end_date = date_filter
        else:
            start_date = end_date = date_filter
        ymd_start, ymd_end = self._to_ymd(start_date), self._to_ymd(end_date)

        # Pull row-level delays; do NOT sum in SQL so we can aggregate flexibly here.
        sql = f"""
        SELECT
            CAST(p.DELAY_DEP AS FLOAT) / 60.0 AS Total_Delay_Minutes,
            p.REAL_DATE_DEP AS time,
            op.Complete_name_in_French AS station,
            op.Geo_Point AS Geo_Point
        FROM punctuality_public AS p
        INNER JOIN operational_points AS op
            ON p.STOPPING_PLACE_ID = op.PTCAR_ID
        WHERE p.DELAY_DEP > 0
          AND p.REAL_DATE_DEP BETWEEN '{ymd_start}' AND '{ymd_end}'
          AND op.Complete_name_in_French IS NOT NULL
        """

        try:
            raw = self.conn.query(sql)
            df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(
                raw, columns=["Total_Delay_Minutes", "time", "station", "Geo_Point"]
            )
        except Exception as e:
            print("Database error occurred:", e)
            self.merged = pd.DataFrame()
            return

        if df.empty:
            self.merged = pd.DataFrame()
            return

        # types + cleanup
        df["Total_Delay_Minutes"] = pd.to_numeric(df["Total_Delay_Minutes"], errors="coerce")
        df["station"] = df["station"].astype("string").str.strip().str.title()
        df["Geo_Point"] = df["Geo_Point"].apply(self._parse_geo_point)
        df = df.dropna(subset=["Total_Delay_Minutes", "Geo_Point", "station"])
        if df.empty:
            self.merged = pd.DataFrame()
            return

        # optional station filter
        if station_filter:
            if isinstance(station_filter, str):
                station_filter = [station_filter]
            df = df[self._casefold_in(df["station"], station_filter)]
            if df.empty:
                self.merged = pd.DataFrame()
                return

        # collapse stacked points (same lat/lon) → one bubble per station
        if self.aggregate:
            func = self._agg_func(self.agg_metric)
            df = (df.groupby(["station", "Geo_Point"], as_index=False)
                    .agg(Total_Delay_Minutes=("Total_Delay_Minutes", func),
                         n=("Total_Delay_Minutes", "size")))
        else:
            # or jitter to separate multiples visually
            if self.jitter_meters > 0:
                df = df.copy()
                df["Geo_Point"] = [self._jitter_point(pt, self.jitter_meters, seed=i)
                                   for i, pt in enumerate(df["Geo_Point"])]

        # small → big so larger bubbles draw on top
        df = df.sort_values("Total_Delay_Minutes")
        self.merged = df[["station", "Geo_Point", "Total_Delay_Minutes"]].reset_index(drop=True)

    def render_map(self):
        if self.merged.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        df = self.merged.copy()
        df["Total_Delay_Minutes"] = pd.to_numeric(df["Total_Delay_Minutes"], errors="coerce")
        df = df.dropna(subset=["Total_Delay_Minutes", "Geo_Point", "station"])
        if df.empty:
            return folium.Map(location=(50.8503, 4.3517), zoom_start=7, tiles="cartodb positron")

        # center
        lats = [pt[0] for pt in df["Geo_Point"]]
        lons = [pt[1] for pt in df["Geo_Point"]]
        m = folium.Map(location=(float(np.mean(lats)), float(np.mean(lons))),
                       zoom_start=7, tiles="cartodb positron")

        # strict linear size mapping
        d = df["Total_Delay_Minutes"].to_numpy(dtype=float)
        dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
        if not math.isfinite(dmin) or not math.isfinite(dmax) or dmax <= dmin:
            dmax = dmin + 1e-6
        sizes = np.interp(d, (dmin, dmax), (self.min_radius, self.max_radius)).astype(float)

        # color: green (low) → red (high)
        span = (dmax - dmin) if (dmax - dmin) != 0 else 1e-6
        def color_for(val: float) -> str:
            t = (val - dmin) / span
            r = int(255 * t); g = int(255 * (1 - t))
            return f"#{r:02x}{g:02x}00"

        for (station, (lat, lon), delay_min, radius) in zip(
            df["station"], df["Geo_Point"], d, sizes
        ):
            folium.CircleMarker(
                location=(float(lat), float(lon)),
                radius=float(radius),
                color=color_for(float(delay_min)),
                fill=True,
                fill_color=color_for(float(delay_min)),
                fill_opacity=0.7,
                popup=f"{station}<br>Delay: {round(float(delay_min), 1)} min"
            ).add_to(m)

        return m

import datetime
import pandas as pd
import plotly.express as px
import re
from typing import Optional, Union, Iterable, Tuple

DateLike  = Union[str, datetime.date, datetime.datetime]
DateRange = Tuple[DateLike, DateLike]

import datetime
import pandas as pd
import plotly.express as px

class DelayHeatmapDB:
    def __init__(self, conn, date_filter):
        """
        :param conn: DB connection/engine
        :param date_filter: datetime.date | datetime.datetime | str
        """
        self.conn = conn
        self.date_filter = pd.to_datetime(date_filter).date()

    def query_delay_data(self, arrival=False, station_filter=None):
        """
        Return columns: delay (minutes), time, station_name, Hour (0..23)
        """
        delay_col = "DELAY_ARR" if arrival else "DELAY_DEP"
        time_col  = "PLANNED_DATETIME_ARR" if arrival else "PLANNED_DATETIME_DEP"

        date_str = self.date_filter.strftime("%Y-%m-%d")

        # Convert delay to MINUTES in SQL to avoid confusion
        sql = f"""
        SELECT 
            CAST(p.{delay_col} AS FLOAT) / 60.0 AS delay,            -- minutes
            p.{time_col} AS time,
            DATEPART(HOUR, p.{time_col}) AS Hour,                    -- 0..23
            op.Complete_name_in_French AS station_name
        FROM punctuality_public AS p
        INNER JOIN operational_points AS op
            ON p.STOPPING_PLACE_ID = op.PTCAR_ID                     -- int-to-int join
        WHERE 
            TRY_CONVERT(DATE, p.{time_col}) = '{date_str}'
            AND p.{delay_col} IS NOT NULL
            AND p.{delay_col} > 0
            AND op.Complete_name_in_French IS NOT NULL
        """

        # Optional station filter
        if station_filter:
            names = [str(s).strip().title().replace("'", "''") for s in station_filter]
            in_clause = ", ".join(f"'{n}'" for n in names)
            sql += f"\n  AND op.Complete_name_in_French IN ({in_clause})"

        # Execute
        df = pd.read_sql_query(sql, self.conn)

        # Clean/enrich
        df["delay"] = pd.to_numeric(df["delay"], errors="coerce")           # already in minutes
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").astype("Int64")
        df["station_name"] = df["station_name"].astype("string").str.strip().str.title()

        df = df.dropna(subset=["delay", "Hour", "station_name"])
        df["Hour"] = df["Hour"].astype(int)

        # Debug (optional)
        # print(df.describe(include='all'))

        return df

    def create_pivot(self, df):
        """
        Pivot: rows=station, cols=Hour (0..23), values=sum of delay (minutes).
        """
        grouped = (
            df.groupby(["station_name", "Hour"], as_index=False)["delay"]
              .sum()  # already in minutes
        )

        pivot = grouped.pivot(index="station_name", columns="Hour", values="delay").fillna(0)

        # Ensure all 24 hours present
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0

        pivot = pivot[sorted(pivot.columns)]
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total").head(10)
        return pivot

    def render_heatmap(self, pivot_table, arrival=False):
        """
        Plotly heatmap; values are minutes.
        """
        title = "Arrival" if arrival else "Departure"
        return px.imshow(
            pivot_table,
            labels=dict(x="Hour", y="Station", color="Total Delay (min)"),
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title=f"{title} Delay Heatmap (Top 10 Stations)"
        )



class DelayLineChart:
    """
    Draws a line chart showing average departure and arrival delays over time,
    filtered by selected stations.
    """

    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.strip().str.title()
        self.df["Actual departure time"] = self._parse_datetime_column(self.df, "Actual departure time")
        self.df["Actual arrival time"] = self._parse_datetime_column(self.df, "Actual arrival time")
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
        return parsed

    def _aggregate_delays(self, station_filter=None, time_unit="hour"):
        df = self.df.copy()

        if station_filter:
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        if time_unit == "hour":
            df["Hour"] = df["Actual departure time"].dt.hour
        elif time_unit == "date":
            df["Hour"] = df["Actual departure time"].dt.date
        else:
            raise ValueError("time_unit must be 'hour' or 'date'")

        agg = df.groupby("Hour").agg({
            "Delay at departure": "mean",
            "Delay at arrival": "mean"
        }).dropna()

        return agg

    def render_line_chart(self, station_filter=None, time_unit="hour"):
        agg = self._aggregate_delays(station_filter=station_filter, time_unit=time_unit)

        fig = go.Figure()

        # Departure Delay in Gold
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at departure"],
            mode='lines+markers',
            name='Departure Delay (min)',
            line=dict(color='gold', width=2),
            marker=dict(color='gold')
        ))

        # Arrival Delay in Orange
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at arrival"],
            mode='lines+markers',
            name='Arrival Delay (min)',
            line=dict(color='red', width=2),
            marker=dict(color='red')
        ))

        fig.update_layout(
            title="📈 Average Delays per Hour (Selected Stations)",
            xaxis_title="Hour of Day" if time_unit == "hour" else "Date",
            yaxis_title="Delay (minutes)",
            xaxis=dict(
                type="category" if time_unit == "hour" else "linear",
                tickmode='linear',
                dtick=1,
                tick0=0,
                tickangle=0
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
            margin=dict(t=50, l=50, r=50, b=50),
            height=450
        )

        return fig






class DelayLineChart:
    """
    Draws a line chart showing average departure and arrival delays over time,
    filtered by selected stations.
    """

    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.strip().str.title()
        self.df["Actual departure time"] = self._parse_datetime_column(self.df, "Actual departure time")
        self.df["Actual arrival time"] = self._parse_datetime_column(self.df, "Actual arrival time")
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
        return parsed

    def _aggregate_delays(self, station_filter=None, time_unit="hour"):
        df = self.df.copy()

        if station_filter:
            df = df[df["Stopping place (FR)"].isin(station_filter)]

        if time_unit == "hour":
            df["Hour"] = df["Actual departure time"].dt.hour
        elif time_unit == "date":
            df["Hour"] = df["Actual departure time"].dt.date
        else:
            raise ValueError("time_unit must be 'hour' or 'date'")

        agg = df.groupby("Hour").agg({
            "Delay at departure": "mean",
            "Delay at arrival": "mean"
        }).dropna()

        return agg

    def render_line_chart(self, station_filter=None, time_unit="hour"):
        agg = self._aggregate_delays(station_filter=station_filter, time_unit=time_unit)

        fig = go.Figure()

        # Departure Delay in Gold
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at departure"],
            mode='lines+markers',
            name='Departure Delay (min)',
            line=dict(color='gold', width=2),
            marker=dict(color='gold')
        ))

        # Arrival Delay in Orange
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg["Delay at arrival"],
            mode='lines+markers',
            name='Arrival Delay (min)',
            line=dict(color='red', width=2),
            marker=dict(color='red')
        ))

        fig.update_layout(
            title="📈 Average Delays per Hour (Selected Stations)",
            xaxis_title="Hour of Day" if time_unit == "hour" else "Date",
            yaxis_title="Delay (minutes)",
            xaxis=dict(
                type="category" if time_unit == "hour" else "linear",
                tickmode='linear',
                dtick=1,
                tick0=0,
                tickangle=0
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
            margin=dict(t=50, l=50, r=50, b=50),
            height=450
        )

        return fig





class DelayHourlyTotalLineChart:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_data()

    def _prepare_data(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.strip().str.title()
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Total Delay"] = self.df["Delay at departure"].fillna(0) + self.df["Delay at arrival"].fillna(0)

        # Combine time fields
        self.df["Hour"] = self.df["Actual departure time"].combine_first(self.df["Actual arrival time"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.df["Hour"] = pd.to_datetime(self.df["Hour"], errors="coerce").dt.hour

        self.df = self.df[self.df["Total Delay"] > 0]

    def plot(self, selected_stations=None, return_data=False):
        df = self.df.copy()
        df["Stopping place (FR)"] = df["Stopping place (FR)"].astype(str).str.title()
        df["Delay at departure"] = pd.to_numeric(df["Delay at departure"], errors="coerce")
        df["Delay at arrival"] = pd.to_numeric(df["Delay at arrival"], errors="coerce")

        # Combine delay columns
        df["Total Delay"] = df["Delay at departure"].fillna(0) + df["Delay at arrival"].fillna(0)
        df = df[df["Total Delay"] > 0]

        # Apply filter
        if selected_stations:
            df = df[df["Stopping place (FR)"].isin(selected_stations)]

        # Extract hour from combined time
        df["Hour"] = df["Actual departure time"].combine_first(df["Actual arrival time"])
        df["Hour"] = pd.to_datetime(df["Hour"], errors="coerce").dt.hour
        df = df.dropna(subset=["Hour"])

        # Group by station and hour
        grouped = (
            df.groupby(["Stopping place (FR)", "Hour"])["Total Delay"]
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={"Total Delay": "Total Delay (min)"})
        )

        if grouped.empty:
            return (None, grouped) if return_data else None

        # Plot
        import plotly.express as px
        fig = px.line(
            grouped,
            x="Hour",
            y="Total Delay (min)",
            color="Stopping place (FR)",
            markers=True,
            title="⏳ Hourly Total Delay by Station"
        )

        # Highlight max point per line
        max_points = grouped.loc[grouped.groupby("Stopping place (FR)")["Total Delay (min)"].idxmax()]
        fig.add_trace(px.scatter(
            max_points,
            x="Hour",
            y="Total Delay (min)",
            color_discrete_sequence=["red"],
            hover_name="Stopping place (FR)",
            hover_data=["Total Delay (min)"]
        ).data[0])

        fig.update_layout(
            xaxis=dict(title="Hour of Day", dtick=1),
            yaxis_title="Delay (minutes)",
            height=450,
            legend_title="Station"
        )

        return (fig, grouped) if return_data else fig






class DelayHourlyTotalLineChartByTrain:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_fields()

    def _prepare_fields(self):
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Total Delay"] = self.df["Delay at departure"].fillna(0) + self.df["Delay at arrival"].fillna(0)

        self.df["Hour"] = self.df["Actual departure time"].combine_first(self.df["Actual arrival time"])
        self.df["Hour"] = pd.to_datetime(self.df["Hour"], errors="coerce").dt.hour

    def plot(self, selected_trains=None, return_data=False):
        df = self.df.copy()
        df = df[df["Total Delay"] > 0]
        df = df.dropna(subset=["Hour"])

        if selected_trains:
            df = df[df["Train number"].isin(selected_trains)]
        
        # Sum delay per hour per train number
        grouped = (
            df.groupby(["Train number", "Hour"])["Total Delay"]
            .sum()
            .div(60)
            .reset_index()
            .rename(columns={"Total Delay": "Total Delay (min)"})
        )

        # Ensure full 24-hour coverage
        all_hours = pd.DataFrame({'Hour': list(range(24))})
        full_data = []
        for train in grouped["Train number"].unique():
            train_data = grouped[grouped["Train number"] == train].merge(all_hours, on="Hour", how="right")
            train_data["Train number"] = train
            train_data["Total Delay (min)"] = train_data["Total Delay (min)"].fillna(0)
            full_data.append(train_data)
        
        grouped_full = pd.concat(full_data, ignore_index=True)

        if grouped_full.empty:
            return (None, grouped_full) if return_data else None

        import plotly.express as px
        fig = px.line(
            grouped_full,
            x="Hour",
            y="Total Delay (min)",
            color="Train number",
            markers=True,
            title="⏳ Hourly Total Delay by Train Number"
        )

        # Max points (after ensuring 24h coverage)
        max_points = grouped_full.loc[grouped_full.groupby("Train number")["Total Delay (min)"].idxmax()]
        fig.add_trace(px.scatter(
            max_points,
            x="Hour",
            y="Total Delay (min)",
            color_discrete_sequence=["red"],
            hover_name="Train number",
            hover_data=["Total Delay (min)"]
        ).data[0])

        fig.update_layout(
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1, range=[0, 23]),
            yaxis_title="Delay (minutes)",
            height=450,
            legend_title="Train Number"
        )

        return (fig, grouped_full) if return_data else fig



class DelayHourlyLinkTotalLineChart:
    def __init__(self, delay_data_path: str):
        self.df = pd.read_csv(delay_data_path)
        self._prepare_common_fields()

    def _prepare_common_fields(self):
        self.df["Stopping place (FR)"] = self.df["Stopping place (FR)"].astype(str).str.title()
        self.df["Train number"] = self.df["Train number"].astype(str)
        self.df["Relation direction"] = self.df["Relation direction"].astype(str)
        self.df["Delay at departure"] = pd.to_numeric(self.df["Delay at departure"], errors="coerce")
        self.df["Delay at arrival"] = pd.to_numeric(self.df["Delay at arrival"], errors="coerce")
        self.df["Actual departure time"] = self._parse_datetime_column(self.df, "Actual departure time")
        self.df["Actual arrival time"] = self._parse_datetime_column(self.df, "Actual arrival time")

    def _parse_datetime_column(self, df: pd.DataFrame, time_col: str) -> pd.Series:
        parsed = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        mask = parsed.isna() & df[time_col].notna()
        if mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[mask] = pd.to_datetime(df.loc[mask, time_col], errors="coerce")
        return parsed

    def _group_by_hour(self, df: pd.DataFrame, group_col: str):
        df["Total Delay"] = df["Delay at departure"].fillna(0) + df["Delay at arrival"].fillna(0)
        df = df[df["Total Delay"] > 0]
        df["Hour"] = df["Actual departure time"].combine_first(df["Actual arrival time"]).dt.hour
        grouped = df.groupby([group_col, "Hour"])["Total Delay"].sum().div(60).reset_index()
        return grouped

    def _plot_grouped(self, grouped: pd.DataFrame, group_col: str):
        if grouped.empty:
            return None
        fig = go.Figure()
        for group in grouped[group_col].unique():
            df_group = grouped[grouped[group_col] == group]
            max_val = df_group["Total Delay"].max()
            fig.add_trace(go.Scatter(
                x=df_group["Hour"],
                y=df_group["Total Delay"],
                mode="lines+markers+text",
                name=group,
                line=dict(width=2),
                text=[f"⬆️ {val:.1f}" if val == max_val else "" for val in df_group["Total Delay"]],
                textposition="top center"
            ))
        fig.update_layout(
            title=f"Total Delay by Hour per {group_col}",
            xaxis_title="Hour of Day",
            yaxis_title="Total Delay (min)",
            xaxis=dict(dtick=1),
            height=500,
            legend_title=group_col
        )
        return fig

    def plot(self, selected_stations=None, return_data=False):
        df = self.df.copy()
        if selected_stations:
            df = df[df["Stopping place (FR)"].isin(selected_stations)]
        grouped = self._group_by_hour(df, "Stopping place (FR)")
        fig = self._plot_grouped(grouped, "Stopping place (FR)")
        return (fig, grouped) if return_data else fig

    def plot_by_train_number(self, selected_trains=None, return_data=False):
        df = self.df.copy()
        if selected_trains:
            df = df[df["Train number"].isin(selected_trains)]
        grouped = self._group_by_hour(df, "Train number")
        fig = self._plot_grouped(grouped, "Train number")
        return (fig, grouped) if return_data else fig

    def plot_by_relation_direction(self, selected_relations=None, return_data=False):
        df = self.df.copy()
        if selected_relations:
            df = df[df["Relation direction"].isin(selected_relations)]
        grouped = self._group_by_hour(df, "Relation direction")
        fig = self._plot_grouped(grouped, "Relation direction")
        return (fig, grouped) if return_data else fig
