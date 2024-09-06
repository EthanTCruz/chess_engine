from google.cloud import storage
import os
import base64
import json
from google.oauth2 import service_account

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID to give your GCS object
    # destination_blob_name = "storage-object-name"
    path_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if path_credentials == '/var/secrets/google/key.json':
        with open(path_credentials, 'r') as file:
            encoded_secret = file.read()

            decoded_secret = base64.b64decode(encoded_secret)
            decoded_secret_str = decoded_secret.decode('utf-8')

            service_account_info = json.loads(decoded_secret_str)
            credentials = service_account.Credentials.from_service_account_info(service_account_info)

            storage_client = storage.Client(credentials=credentials)
    else:
        storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
