from azure import identity
import pyodbc, struct, os
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, col
import pandas as pd
from pyspark.sql import SparkSession
import duckdb
import threading

_shared_connector = None
_connector_lock = threading.Lock()

class DBConnector:
    def __init__(self):  
        self.conn_string = os.getenv('DB_CONN_PROD')
        if self.conn_string is None:
            raise ValueError("DB_CONN_PROD environment variable is not set")
        self.conn = pyodbc.connect(self.conn_string)
        
    def get_token(self):
        credential = identity.DefaultAzureCredential(exclude_interactive_browser_credential=False)
        token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        return token_struct,token_bytes
        
    
    def query(self, query, params=None) -> list[dict]:
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            query_type = query.strip().lower()
            if query_type.startswith(('insert', 'update', 'delete')):
                self.conn.commit()
                cursor.close()
                return None

            columns = [column[0] for column in cursor.description]
            results = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in results]
            cursor.close()
            return data

        except pyodbc.Error as e:
            print(f"Database error occurred: {str(e)}")
            print(query)
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print(query)
            raise

    def get_table_columns(self, table_name):
        return self.query(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",[table_name])


    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()

def get_db_connector():
    global _shared_connector
    if _shared_connector is None:
        with _connector_lock:
            if _shared_connector is None:
                _shared_connector = DBConnector()
    return _shared_connector