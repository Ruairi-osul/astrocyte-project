import boto3
import os
from pathlib import Path


DATA_DIR = os.environ.get("OUT_DIR", "data")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
PREFIX = "dataset-03"
S3_REGION = "us-east-1"


def construct_local_path(s3_key, local_root_dir):
    """
    Constructs a local file path from the S3 key.
    """
    local_path = Path(local_root_dir) / Path(s3_key)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path


class S3BucketDownloader:
    def __init__(self, bucket_name, prefix, file_format):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.file_format = file_format
        self.s3 = boto3.client("s3")

    def list_files(self):
        """
        List all files in the bucket with the specified prefix and file format.
        """
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        all_files = response.get("Contents", [])
        return [
            file["Key"] for file in all_files if file["Key"].endswith(self.file_format)
        ]

    def download_files(self, download_path):
        """
        Download files from the S3 bucket, preserving the directory structure.
        """
        files = self.list_files()
        for file_key in files:
            local_file_path = construct_local_path(file_key, download_path)
            self.s3.download_file(self.bucket_name, file_key, str(local_file_path))
        print(f"Downloaded {len(files)} files to {download_path}")


def main():
    local_dir = Path(DATA_DIR) / PREFIX
    local_dir.mkdir(parents=True, exist_ok=True)

    # downloader_parquet = S3BucketDownloader(S3_BUCKET, PREFIX, ".parquet")
    # downloader_parquet.download_files(DATA_DIR)

    # downloader_csv = S3BucketDownloader(S3_BUCKET, PREFIX, ".csv")
    # downloader_csv.download_files(DATA_DIR)

    downloader_pkl = S3BucketDownloader(S3_BUCKET, PREFIX, ".pkl")
    downloader_pkl.download_files(DATA_DIR)

if __name__ == "__main__":
    main()