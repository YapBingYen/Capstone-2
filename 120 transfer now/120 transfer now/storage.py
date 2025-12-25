import os
import io
import shutil
from typing import Optional


class StorageError(Exception):
    pass


class LocalStorage:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def upload_fileobj(self, fileobj, key: str) -> str:
        dest_path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            shutil.copyfileobj(fileobj, f)
        return dest_path

    def move(self, src_path: str, dest_key: str) -> str:
        dest_path = os.path.join(self.base_dir, dest_key)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)
        return dest_path

    def public_url(self, path_or_key: str) -> Optional[str]:
        return None


class S3Storage:
    def __init__(self, bucket: str, region: str, prefix: str = ""):
        import boto3
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip('/')
        self.s3 = boto3.client('s3', region_name=region)

    def _key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key.lstrip('/')}"
        return key.lstrip('/')

    def upload_fileobj(self, fileobj, key: str) -> str:
        k = self._key(key)
        self.s3.upload_fileobj(fileobj, self.bucket, k, ExtraArgs={"ContentType": "image/jpeg", "ACL": "public-read"})
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{k}"

    def move(self, src_url_or_key: str, dest_key: str) -> str:
        k_src = src_url_or_key
        # If a full URL is provided, extract the key after bucket host
        if 'amazonaws.com/' in k_src:
            k_src = k_src.split('amazonaws.com/', 1)[-1]
        k_src = self._key(k_src)
        k_dst = self._key(dest_key)
        self.s3.copy_object(Bucket=self.bucket, CopySource={'Bucket': self.bucket, 'Key': k_src}, Key=k_dst, ACL='public-read')
        self.s3.delete_object(Bucket=self.bucket, Key=k_src)
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{k_dst}"

    def public_url(self, key: str) -> str:
        k = self._key(key)
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{k}"


def get_storage():
    backend = os.environ.get('STORAGE_BACKEND', 'local').lower()
    if backend == 's3':
        bucket = os.environ.get('AWS_S3_BUCKET', '')
        region = os.environ.get('AWS_REGION', 'ap-southeast-1')
        prefix = os.environ.get('AWS_S3_PREFIX', 'pet-id')
        if not bucket:
            raise StorageError('AWS_S3_BUCKET is required for S3 storage')
        return S3Storage(bucket=bucket, region=region, prefix=prefix)
    # local fallback: base dir is project root data folder
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    return LocalStorage(base_dir)
