import os
import sys

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_fetcher.cloud_utils import StorageHandler

def main():
    load_dotenv()
    # Load credentials from environment variables
    endpoint_url = os.getenv('FILEBASE_ENDPOINT', 'https://s3.filebase.com')
    access_key = os.getenv('FILEBASE_ACCESS_KEY')
    secret_key = os.getenv('FILEBASE_SECRET_KEY')
    bucket_name = os.getenv('FILEBASE_BUCKET')
    
    # Local fallback base directory
    local_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    storage = StorageHandler(endpoint_url, access_key, secret_key, bucket_name, local_base=local_base)

    # Define the specific files to fetch
    target_files = [
        "merged/features/crypto_features.parquet",
        "merged/features/stocks_features.parquet", 
        "merged/train/crypto_features_train.parquet",
        "merged/train/stocks_features_train.parquet",
        "merged/features/crypto_report.json",
        "merged/features/stocks_report.json"
    ]

    print(f"Fetching specific files from S3...")
    
    for key in target_files:
        try:
            # Download the file
            data = storage.download(key)
            
            # Remove 'merged/' from the start of the key for local path
            local_rel_path = key[len("merged/"):] if key.startswith("merged/") else key
            local_path = os.path.join(local_base, local_rel_path)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Write the file
            with open(local_path, 'wb') as f:
                f.write(data)
            
            print(f"[OK] Saved {key} to local path: {local_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch {key}: {e}")

    print("Fetch operation completed.")

if __name__ == "__main__":
    main()