import os, sys
from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv
from io import BytesIO
import tempfile
from pyspark.sql import DataFrame, SparkSession
from typing import Any

class LakeReader:
    """
    A class for reading data files from Azure Data Lake Storage using PySpark.
    
    This class provides methods to download and read various file formats (CSV, Parquet)
    from Azure Data Lake Storage, converting them to Spark DataFrames.
    
    Attributes:
        spark (SparkSession): The Spark session for DataFrame operations
        connection_string (str): Azure Blob Storage connection string
        loaded_data (dict): Cache of loaded DataFrames indexed by blob/filename
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the LakeReader with a Spark session.
        
        Args:
            spark (SparkSession): The Spark session for DataFrame operations
            
        Note:
            Loads Azure Blob connection string from environment variables.
        """
        load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))
        self.spark = spark
        self.connection_string = os.getenv("AZURE_BLOB_CONN")
        self.loaded_data = {}

    def download_file(self, blob: str, filename: str) -> bytes:
        """
        Download a file from Azure Data Lake Storage.
        
        Args:
            blob (str): The blob container name
            filename (str): The name of the file to download
            
        Returns:
            bytes: The downloaded file content as bytes
            
        Raises:
            Exception: If download fails
        """
        try:
            service_client = DataLakeServiceClient.from_connection_string(self.connection_string)
            file_system_client = service_client.get_file_system_client(blob)
            file_client = file_system_client.get_file_client(filename)
            download = file_client.download_file()
            downloaded_bytes = download.readall()
            return downloaded_bytes
        except Exception as e:
            print(f"Erreur lors du téléchargement du fichier {blob}/{filename}: {str(e)}")
            raise
    
    def create_temp_file(self, downloaded_bytes: bytes, blob: str, filename: str, suffix: str) -> str:
        """
        Create a temporary file from downloaded bytes.
        
        Args:
            downloaded_bytes (bytes): The file content as bytes
            blob (str): The blob container name (for reference)
            filename (str): The original filename (for reference)
            suffix (str): The file extension suffix (e.g., '.csv', '.parquet')
            
        Returns:
            str: The path to the created temporary file
            
        Raises:
            Exception: If temporary file creation fails
        """
        try:
            # Convert the downloaded bytes to a Spark DataFrame
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(downloaded_bytes)
                temp_file_path = temp_file.name
            return temp_file_path
        except Exception as e:
            print(f"Erreur lors de la création du fichier temporaire: {str(e)}")
            raise
    
    def read_csv(self, blob: str, filename: str, sep=';') -> DataFrame:
        """
        Read a CSV file from Azure Data Lake Storage into a Spark DataFrame.
        
        Args:
            blob (str): The blob container name
            filename (str): The name of the CSV file
            sep (str, optional): The delimiter character. Defaults to ';'.
            
        Returns:
            DataFrame: Spark DataFrame containing the CSV data
            
        Raises:
            FileNotFoundError: If the temporary file doesn't exist
            Exception: If reading the CSV file fails
        """
        try:
            downloaded_bytes = self.download_file(blob, filename)
            temp_file_path = self.create_temp_file(downloaded_bytes, blob, filename, ".csv")

             # Vérifier que le fichier existe avant de le lire
            if not os.path.exists(temp_file_path):
                raise FileNotFoundError(f"Le fichier temporaire {temp_file_path} n'existe pas")
            
            df = self.spark.read.csv(temp_file_path, header=True, inferSchema=True, sep=sep)

            #self.loaded_data[blob + "/" + filename] = df
           
            #os.remove(temp_file_path)
            return df
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier CSV {blob}/{filename}: {str(e)}")
            #os.remove(temp_file_path)
            raise
    
    def read_parquet(self, blob: str, filename: str) -> DataFrame:
        """
        Read a Parquet file from Azure Data Lake Storage into a Spark DataFrame.
        
        Args:
            blob (str): The blob container name
            filename (str): The name of the Parquet file
            
        Returns:
            DataFrame: Spark DataFrame containing the Parquet data
            
        Raises:
            FileNotFoundError: If the temporary file doesn't exist
            Exception: If reading the Parquet file fails
            
        Note:
            The DataFrame is cached in loaded_data for potential reuse.
        """
        try:
            downloaded_bytes = self.download_file(blob, filename)
            temp_file_path = self.create_temp_file(downloaded_bytes, blob, filename, ".parquet")
            
            # Vérifier que le fichier existe avant de le lire
            if not os.path.exists(temp_file_path):
                raise FileNotFoundError(f"Le fichier temporaire {temp_file_path} n'existe pas")
            
            # Lire le fichier Parquet
            df = self.spark.read.parquet(temp_file_path)
            
            # Stocker le DataFrame en mémoire
            self.loaded_data[blob + "/" + filename] = df
            
            return df
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier Parquet {blob}/{filename}: {str(e)}")
            #os.remove(temp_file_path)
            raise