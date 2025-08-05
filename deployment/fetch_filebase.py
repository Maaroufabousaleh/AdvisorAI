import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_cloud.cloud_utils import StorageHandler

def main():
    load_dotenv()
    # Load credentials from environment variables
    endpoint_url = os.getenv('FILEBASE_ENDPOINT', 'https://s3.filebase.com')
    access_key = os.getenv('FILEBASE_ACCESS_KEY')
    secret_key = os.getenv('FILEBASE_SECRET_KEY')
    bucket_name = os.getenv('FILEBASE_BUCKET')
    # Local fallback base directory
    local_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'merged'))

    storage = StorageHandler(endpoint_url, access_key, secret_key, bucket_name, local_base=local_base)

    # Fetch all folders/files from advisorai-data
    advisor_prefix = "advisorai-data/"
    print(f"Fetching all folders/files from: {advisor_prefix}")
    advisor_keys = []
    if storage.s3 and bucket_name:
        try:
            resp = storage.s3.list_objects_v2(Bucket=bucket_name, Prefix=advisor_prefix)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                if not key.endswith('/'):
                    advisor_keys.append(key)
        except Exception as e:
            print(f"[WARN] Could not list objects for {advisor_prefix}: {e}")
    else:
        print(f"[ERROR] No S3 client or bucket configured for advisorai-data!")
    # Download advisorai-data files
    for key in advisor_keys:
        try:
            data = storage.download(key)
            # Remove 'advisorai-data/' from the start of the key for local path
            local_rel_path = key[len("advisorai-data/"):] if key.startswith("advisorai-data/") else key
            local_path = os.path.join(local_base, '..', 'advisorai-data', local_rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)
            print(f"[OK] Saved advisorai-data file locally: {local_path}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch advisorai-data file {key}: {e}")


    # Fetch everything under merged/ except only the last 7 from merged/archive/
    merged_prefix = "merged/"
    print(f"Fetching everything under: {merged_prefix} (except only last 7 from archive)")
    merged_keys = []
    archive_prefix = "merged/archive/"
    archive_folders = set()
    archive_keys = []
    if storage.s3 and bucket_name:
        try:
            resp = storage.s3.list_objects_v2(Bucket=bucket_name, Prefix=merged_prefix)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                # Exclude all archive keys for now
                if key.startswith(archive_prefix):
                    # Collect archive folders for later
                    parts = key[len(archive_prefix):].split('/')
                    if len(parts) > 1 and parts[0].isdigit():
                        archive_folders.add(parts[0])
                    continue
                if not key.endswith('/'):
                    merged_keys.append(key)
        except Exception as e:
            print(f"[WARN] Could not list objects for {merged_prefix}: {e}")
    else:
        print(f"[ERROR] No S3 client or bucket configured for merged!")

    # Download all merged/ (except archive)
    for key in merged_keys:
        try:
            data = storage.download(key)
            local_rel_path = key[len("merged/"):] if key.startswith("merged/") else key
            local_path = os.path.join(local_base, local_rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)
            print(f"[OK] Saved locally: {local_path}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch {key}: {e}")

    # Fetch only the last 7 folders under merged/archive
    archive_prefix = "merged/archive/"
    print(f"Fetching last 7 archive folders from: {archive_prefix}")
    archive_folders = set()
    archive_keys = []
    if storage.s3 and bucket_name:
        try:
            resp = storage.s3.list_objects_v2(Bucket=bucket_name, Prefix=archive_prefix)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                # Expect keys like merged/archive/YYYYMMDD/...
                parts = key[len(archive_prefix):].split('/')
                if len(parts) > 1 and parts[0].isdigit():
                    archive_folders.add(parts[0])
            # Sort and get last 7 folders
            last7 = sorted(archive_folders)[-7:]
            print(f"[INFO] Last 7 archive folders: {last7}")
            # Collect all keys in those folders
            for obj in resp.get('Contents', []):
                key = obj['Key']
                parts = key[len(archive_prefix):].split('/')
                if len(parts) > 1 and parts[0] in last7:
                    archive_keys.append(key)
        except Exception as e:
            print(f"[WARN] Could not list objects for {archive_prefix}: {e}")
    else:
        print(f"[ERROR] No S3 client or bucket configured for archive!")
    # Download archive files
    for key in archive_keys:
        try:
            data = storage.download(key)
            local_rel_path = key[len("merged/"):] if key.startswith("merged/") else key
            local_path = os.path.join(local_base, local_rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)
            print(f"[OK] Saved archive file locally: {local_path}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch archive file {key}: {e}")

if __name__ == "__main__":
    main()
