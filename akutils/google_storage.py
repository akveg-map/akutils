# Define a function to list all files (optionally of a particular extension) in a Google Storage folder
def gcs_list_files(gcs_folder, storage_client, extension=''):
    bucket_name = gcs_folder.split('/')[2]
    prefix = '/'.join(gcs_folder.split('/')[3:])
    # Safely ensure the prefix acts as a folder search by appending a trailing slash if missing
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    # List blobs in storage folder
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    # Identify the full uri for each matching file
    if extension == '':
        file_list = [f'gs://{bucket_name}/{b.name}' for b in blobs if not b.name.endswith('/')]
    else:
        file_list = [f'gs://{bucket_name}/{b.name}' for b in blobs if b.name.endswith(extension)]
    return file_list

# Define function to check if a file exists in GCS
def gcs_blob_exists(gcs_uri, storage_client):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    return bucket.blob(blob_name).exists()

# Define function to download a file from GCS to local storage
def download_from_gcs(gcs_uri, local_path, storage_client):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

# Define function to upload a file from local storage to GCS
def upload_to_gcs(local_path, gcs_uri, storage_client):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Set chunk size to 5 MB
    blob.chunk_size = 5 * 1024 * 1024
    # Upload with one hour timeout
    blob.upload_from_filename(local_path, timeout=3600)

# Define a function to generate vsigs paths for tif rasters
def get_vsi_paths(bucket_name, prefix, storage_client):
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [f'/vsigs/{bucket_name}/{b.name}' for b in blobs if b.name.endswith('.tif')]
