"""
cloud_utils.py – Unified utilities for HTTP fetch and cloud/local storage operations.

Provides:
  • fetch_content / fetch_json for HTTP GET
  • StorageHandler class with upload/download and fallback to local filesystem
    - Methods set self.last_mode to 'cloud' or 'local'
    - Local files are stored under a base directory

Usage:
  from cloud_utils import StorageHandler, fetch_json

Requirements:
  • boto3 and botocore
  • requests
  • ENV vars for cloud credentials (e.g. FILEBASE_*)
"""
import os
import errno
import requests
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# HTTP Fetch utilities ---------------------------------------------------------
def fetch_content(url, headers=None, timeout=15):
    """Fetch binary content via HTTP GET."""
    resp = requests.get(url, headers=headers, timeout=timeout, stream=False)
    resp.raise_for_status()
    return resp.content

def fetch_json(url, headers=None, timeout=15):
    """Fetch JSON data via HTTP GET."""
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", data) if isinstance(data, dict) else data

def fetch_text(url, headers=None, timeout=15, encoding='utf-8'):
    """Fetch text content via HTTP GET."""
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = encoding
    return resp.text

# Storage Handler ---------------------------------------------------------------
class StorageHandler:
    def list_prefix(self, prefix):
        """List all object keys in the given S3 prefix. Returns a list of keys. Local fallback returns empty list."""
        if self.s3 and self.bucket:
            paginator = self.s3.get_paginator('list_objects_v2')
            keys = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    keys.append(obj['Key'])
            return keys
        # Local fallback: not implemented (could walk local filesystem if needed)
        return []
    def __init__(self, endpoint_url, access_key, secret_key, bucket_name, local_base="data"):
        """
        Initialize cloud storage client and local base path.
        endpoint_url: S3-compatible endpoint URL
        bucket_name: target bucket name (if None/empty, operate in local-only mode)
        local_base: directory prefix for local fallback files
        """
        self.bucket = bucket_name
        self.local_base = local_base.rstrip(os.sep)
        self.last_mode = None  # 'cloud' or 'local'
        if bucket_name:
            # boto3 client config
            cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
            self.s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=cfg,
                region_name='us-east-1'
            )
        else:
            self.s3 = None

    def _ensure_local_dir(self, key):
        path = os.path.join(self.local_base, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def download(self, key):
        """Download object by key. Returns bytes, sets last_mode. Raises FileNotFoundError if not found."""
        if self.s3 and self.bucket:
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                data = resp['Body'].read()
                self.last_mode = 'cloud'
                print(f"[OK] Downloaded {key} from s3://{self.bucket}/{key}")
                return data
            except (ClientError, BotoCoreError) as e:
                print(f"[WARN] Could not download {key} from S3: {e}")
        # Always fallback to local if S3 is not configured or download fails
        local_path = self._ensure_local_dir(key)
        try:
            with open(local_path, 'rb') as f:
                data = f.read()
            self.last_mode = 'local'
            print(f"[FALLBACK] Loaded {key} from local {local_path}")
            return data
        except FileNotFoundError:
            print(f"[ERROR] {key} not found in S3 or locally at {local_path}")
            raise

    def upload(self, key, data, content_type='application/octet-stream'):
        """Upload bytes to cloud, fallback to local. Sets last_mode. Returns True if cloud, False if local."""
        if self.s3 and self.bucket:
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
                self.last_mode = 'cloud'
                print(f"[OK] Uploaded {key} -> s3://{self.bucket}/{key}")
                return True
            except (ClientError, BotoCoreError) as e:
                print(f"[ERROR] Failed uploading {key}: {e}")
        # Always fallback to local if S3 is not configured or upload fails
        local_path = self._ensure_local_dir(key)
        with open(local_path, 'wb') as f:
            f.write(data)
        self.last_mode = 'local'
        print(f"[FALLBACK] Saved {key} locally -> {local_path}")
        return False

    def exists(self, key):
        """Check for existence of object. Returns True if found in cloud or local."""
        if self.s3 and self.bucket:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except (ClientError, BotoCoreError):
                pass
        local_path = os.path.join(self.local_base, key)
        return os.path.exists(local_path)

    def delete(self, key):
        """Delete object in cloud or local fallback."""
        if self.s3 and self.bucket:
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
                self.last_mode = 'cloud'
                print(f"[OK] Deleted {key} from s3://{self.bucket}/{key}")
                return
            except Exception:
                pass
        local_path = os.path.join(self.local_base, key)
        try:
            os.remove(local_path)
            self.last_mode = 'local'
            print(f"[FALLBACK] Deleted {key} locally -> {local_path}")
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def get_last_mode(self):
        """Return 'cloud' or 'local' depending on last operation."""
        return self.last_mode
