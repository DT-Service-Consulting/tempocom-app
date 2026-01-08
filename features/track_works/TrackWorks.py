from tempocom.services import DBConnector
import pandas as pd
from datetime import datetime
from typing import Optional
class TrackWorks:
    sql_query = """
    SELECT * FROM colt 
    WHERE status = 'Y'
    """
    
    def __init__(self, dbc: DBConnector, date: Optional[datetime] = None):
        self.dbc = dbc
        if date is None:
            date = datetime.now()
        self.sql_query = self.sql_query + f" AND date_begin <= '{date}' AND date_end >= '{date}'"
        self.df = pd.read_sql_query(self.sql_query, self.dbc.conn)
        self.date_range = self.get_date_range()
    
    def get_date_range(self):
        query = "SELECT MIN(date_begin) as min_date, MAX(date_begin) as max_date FROM colt WHERE status = 'Y'"
        result = pd.read_sql_query(query, self.dbc.conn)
        if result.empty:
            return None, None
        return pd.to_datetime(result.iloc[0]['min_date']).date(), pd.to_datetime(result.iloc[0]['max_date']).date()
    
    def get_active_works(self):
        return self.df
    
    def get_work_by_id(self, work_id):
        work = self.df[self.df['id'] == work_id]
        if work.empty:
            return None
        return work.iloc[0].to_dict()
    
    def has_valid_sections(self, row):
        return not (pd.isna(row['section_from_id']) or pd.isna(row['section_to_id']))
    
    def is_point_work(self, row):
        return row['section_from_id'] == row['section_to_id']

    def get_description(self, id):
        res = pd.read_sql_query(f"SELECT description_of_works FROM colt_descriptions WHERE cou_id = {id}", self.dbc.conn)
        return "No description available" if res.empty else res.iloc[0]['description_of_works']

    def get_descriptions(self, ids: list[int]):
        res = pd.read_sql_query(f"SELECT cou_id, description_of_works FROM colt_descriptions WHERE cou_id IN ({','.join(map(str, ids))})", self.dbc.conn)
        return dict(zip(res['cou_id'], res['description_of_works']))

