from azure import identity
import pyodbc, struct, os
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, col
import pandas as pd
from pyspark.sql import SparkSession

class DBConnector:
    """
    A database connector class for Azure SQL Database using Azure AD authentication.
    
    This class provides methods to connect to Azure SQL Database using Azure AD tokens,
    execute queries, and perform data operations with PySpark DataFrames.
    
    Attributes:
        connection_string (str): The connection string for Azure SQL Database
        conn (pyodbc.Connection): The active database connection
        spark: The Spark session for DataFrame operations
    """
    
    def __init__(self, connection_string=None):
        """
        Initialize the DBConnector with Spark session and optional connection string.
        
        Args:
            spark: The Spark session for DataFrame operations
            connection_string (str, optional): Azure SQL connection string. 
                If None, loads from environment variable AZURE_SQL_CONN
        """
        
        if connection_string is not None:
            self.connection_string = connection_string
        else:
            load_dotenv('../config/.env')
            self.connection_string = os.getenv('AZURE_SQL_CONN')
            self.connection_string_with_pwd = os.getenv('AZURE_ODBC_SQL_CONN')
        self.conn = self.get_conn()

    def get_token(self):
        credential = identity.DefaultAzureCredential(exclude_interactive_browser_credential=False)
        token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("UTF-16-LE")
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        return token_struct,token_bytes
        
    def get_conn(self) -> pyodbc.Connection:
        """
        Establish a connection to Azure SQL Database using Azure AD authentication.
        
        Returns:
            pyodbc.Connection: An active database connection
            
        Note:
            Uses Azure DefaultAzureCredential to obtain access tokens for authentication.
        """
        
        #token_struct,token_bytes = self.get_token()
        SQL_COPT_SS_ACCESS_TOKEN = 1256  # This connection option is defined by microsoft in msodbcsql.h
        #conn = pyodbc.connect(self.connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})
        conn = pyodbc.connect(self.connection_string_with_pwd)
        self.conn = conn
        return conn
    
    def query(self, query, params=None) -> list[dict]:
        """
        Execute a SQL query and return results as a Spark DataFrame.
        
        Args:
            query (str): The SQL query to execute
            params (tuple, optional): Parameters for parameterized queries
            
        Returns:
            DataFrame or None: Spark DataFrame for SELECT queries, None for other operations
            
        Raises:
            pyodbc.Error: If a database error occurs
            Exception: For other unexpected errors
        """
        try:
            if self.conn is None:
                self.get_conn()
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Commit if needed
            if query.strip().lower().startswith(('insert', 'update', 'delete')):
                self.conn.commit()
                return None

            if query.strip().lower().startswith('select'):
                # Close any existing cursors before executing new query
                cursor.close()
                cursor = self.conn.cursor()
                cursor.execute(query)
                
                # Get column names directly from cursor description
                columns = [column[0] for column in cursor.description]
                results = cursor.fetchall()
                cursor.close()
                
                # Convert results to list of dictionaries
                data = [dict(zip(columns, row)) for row in results]
                return data

            return None
        except pyodbc.Error as e:
            print(f"Database error occurred: {str(e)}")
            print(query)
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print(query)
            raise

    def get_table_columns(self, table_name):
        """
        Get the column names of a specified table.
        
        Args:
            table_name (str): The name of the table
            
        Returns:
            list: List of column names in the table
        """
        cursor = self.conn.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
        tc = cursor.fetchall()
        liste_colonnes = [colonne[0] for colonne in tc]
        self.conn.commit()
        return liste_colonnes
        

    def close(self):
        """
        Close the database connection and set it to None.
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """
        Destructor method to ensure the connection is closed when the object is destroyed.
        """
        self.close()