import os, sys
from azure.storage.filedatalake import DataLakeServiceClient
from dotenv import load_dotenv
from io import BytesIO
import tempfile
from pyspark.sql import DataFrame, SparkSession
from typing import Any
class LakeReader:

    def __init__(self, spark: SparkSession):
        load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))
        self.spark = spark
        self.connection_string = os.getenv("AZURE_BLOB_CONN")
        self.loaded_data = {}

    def download_file(self, blob: str, filename: str) -> DataFrame:
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