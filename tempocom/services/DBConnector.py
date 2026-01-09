from azure import identity
import pyodbc, struct, os
import time
import logging

import threading

_shared_connector = None
_connector_lock = threading.Lock()

class DBConnector:
    def __init__(self):  
        self.conn_string = os.getenv('DB_CONN_PROD')
        if self.conn_string is None:
            raise ValueError("DB_CONN_PROD environment variable is not set")
        self.conn = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self._connect()
        
    def _connect(self):
        """Establish database connection with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if self.conn:
                    self.conn.close()
                self.conn = pyodbc.connect(self.conn_string)
                logging.info("Database connection established successfully")
                return
            except pyodbc.Error as e:
                logging.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logging.error("All connection attempts failed")
                    raise
    
    def _ensure_connection(self):
        """Ensure the connection is alive, reconnect if necessary"""
        try:
            if self.conn is None:
                self._connect()
                return
            
            # Test the connection with a simple query
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        except (pyodbc.Error, AttributeError):
            logging.warning("Connection lost, attempting to reconnect...")
            self._connect()

    def get_token(self):
        credential = identity.DefaultAzureCredential(exclude_interactive_browser_credential=False)
        token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        return token_struct,token_bytes
        
    
    def query(self, query, params=None) -> list[dict]:
        for attempt in range(self.max_retries):
            try:
                self._ensure_connection()
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
                error_code = e.args[0] if e.args else None
                logging.error(f"Database error occurred (attempt {attempt + 1}): {str(e)}")
                logging.error(f"Query: {query}")
                
                # Check if it's a connection-related error
                if error_code in ['08S01', '08003', '08007', 'HYT00'] and attempt < self.max_retries - 1:
                    logging.warning("Connection error detected, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
            except Exception as e:
                logging.error(f"An unexpected error occurred: {str(e)}")
                logging.error(f"Query: {query}")
                raise
        
        # If we get here, all retries failed
        raise pyodbc.Error("All query retry attempts failed")

    def get_table_columns(self, table_name):
        return self.query(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",[table_name])
    
    def execute_pandas_query(self, query, params=None):
        """Execute a query and return results suitable for pandas DataFrame"""
        import pandas as pd
        
        for attempt in range(self.max_retries):
            try:
                self._ensure_connection()
                return pd.read_sql_query(query, self.conn, params=params)
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Pandas query error (attempt {attempt + 1}): {error_msg}")
                logging.error(f"Query: {query}")
                
                # Check if it's a connection-related error
                if any(code in error_msg for code in ['08S01', '08003', '08007', 'HYT00', 'Communication link failure']) and attempt < self.max_retries - 1:
                    logging.warning("Connection error detected in pandas query, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        # If we get here, all retries failed
        raise Exception("All pandas query retry attempts failed")


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
    else:
        # Check if existing connection is still alive
        try:
            _shared_connector._ensure_connection()
        except Exception as e:
            logging.warning(f"Shared connector failed health check, creating new one: {str(e)}")
            with _connector_lock:
                _shared_connector = DBConnector()
    return _shared_connector