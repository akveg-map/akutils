# Define function to check if a file exists in GCS
def gcs_blob_exists(gcs_uri):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    return bucket.blob(blob_name).exists()

# Define function to download a file from GCS to local storage
def download_from_gcs(gcs_uri, local_path):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

# Define function to upload a file from local storage to GCS
def upload_to_gcs(local_path, gcs_uri):
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

# Define a function to generate vsigs paths
def get_vsi_paths(bucket_name, prefix):
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [f'/vsigs/{bucket_name}/{b.name}' for b in blobs if b.name.endswith('.tif')]