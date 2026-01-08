import pandas as pd
from tempocom.services import DBConnector
import datetime

class Punctuality:

    def __init__(self, dbc:DBConnector=DBConnector()):
        self.df = pd.DataFrame()

    def filter_by_datetime(self, _from: datetime.datetime, _to: datetime.datetime, dbc:DBConnector=DBConnector()):
        self.df = pd.read_sql_query(f"""
            SELECT *
            FROM punctuality_public
            WHERE 
                TRAIN_NO IN (
                    SELECT DISTINCT TRAIN_NO
                    FROM punctuality_public
                    WHERE 
                        PLANNED_DATETIME_ARR >= '{_from}' AND
                        PLANNED_DATETIME_DEP <= '{_to}' AND
                        REAL_DATE_DEP = '{_from.strftime('%Y-%m-%d')}'
                )
                AND REAL_DATE_DEP = '{_from.strftime('%Y-%m-%d')}'
        """, dbc.conn)
        self.df = self.apply_types()
        return self.df

    def apply_types(self):
        self.df['TRAIN_NO'] = self.df['TRAIN_NO'].astype(int)
        self.df['STOPPING_PLACE_ID'] = self.df['STOPPING_PLACE_ID'].astype(int)
        return self.df

    def compute_real_datetime(self):
        self.df['PLANNED_DATETIME_ARR'] = pd.to_datetime(self.df['PLANNED_DATETIME_ARR'])
        self.df['PLANNED_DATETIME_DEP'] = pd.to_datetime(self.df['PLANNED_DATETIME_DEP'])
        
        self.df['REAL_DATETIME_ARR'] = self.df['PLANNED_DATETIME_ARR'] + pd.to_timedelta(self.df['DELAY_ARR'], unit='s')
        self.df['REAL_DATETIME_DEP'] = self.df['PLANNED_DATETIME_DEP'] + pd.to_timedelta(self.df['DELAY_DEP'], unit='s')
        
        return self.df

    def compute_next_stopping_place(self):
        self.df['NEXT_STOPPING_PLACE_ID'] = self.df.groupby('TRAIN_NO')['STOPPING_PLACE_ID'].shift(-1)
        self.df['NEXT_STOPPING_PLACE_ID'] = self.df['NEXT_STOPPING_PLACE_ID'].fillna(0).astype(int)
        return self.df