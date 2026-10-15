"""S3-compatible object storage (MinIO) helpers."""

import logging
import math

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from app import config

logger = logging.getLogger("dam.storage")

BUCKET = config.MINIO_BUCKET
# Multipart uploads in 16 MiB parts with bounded concurrency: memory stays flat for any file size.
TRANSFER_CONFIG = TransferConfig(multipart_chunksize=16 * 2**20, max_concurrency=4)


def _client(endpoint: str):
    access_key, secret_key = config.minio_credentials()
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
        region_name="us-east-1",
    )


s3 = _client(config.MINIO_ENDPOINT)
# Presigned URLs embed the host in the signature, so they must be signed for the address
# the browser/CLI will use, not the internal Docker hostname.
s3_public = _client(config.MINIO_PUBLIC_ENDPOINT)


def ensure_bucket() -> None:
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        logger.info("Creating bucket '%s'", BUCKET)
        s3.create_bucket(Bucket=BUCKET)


def upload_stream(fileobj, key: str, content_type: str) -> None:
    s3.upload_fileobj(fileobj, BUCKET, key, ExtraArgs={"ContentType": content_type}, Config=TRANSFER_CONFIG)


def download(key: str, path: str) -> None:
    s3.download_file(BUCKET, key, path, Config=TRANSFER_CONFIG)


def head(key: str) -> dict:
    return s3.head_object(Bucket=BUCKET, Key=key)


def read_head(key: str, n: int = 64) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=0-{n - 1}")["Body"].read()


def presigned_get(key: str, content_type: str, filename: str) -> str:
    return s3_public.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": key,
            "ResponseContentType": content_type,
            "ResponseContentDisposition": f'inline; filename="{filename}"',
        },
        ExpiresIn=config.PRESIGNED_URL_TTL_SECONDS,
    )


def delete_object(key: str) -> None:
    s3.delete_object(Bucket=BUCKET, Key=key)


def delete_all() -> int:
    """Deletes every object in the bucket (paginated; delete_objects takes <= 1000 keys)."""
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
            deleted += len(keys)
    # Abandoned direct uploads hold storage until aborted.
    for upload in s3.list_multipart_uploads(Bucket=BUCKET).get("Uploads", []):
        s3.abort_multipart_upload(Bucket=BUCKET, Key=upload["Key"], UploadId=upload["UploadId"])
    return deleted


# -------------------------------------------------------------- presigned multipart uploads
def plan_parts(size_bytes: int, part_mb: int = config.DIRECT_UPLOAD_PART_MB) -> tuple[int, int]:
    """Returns (part_size, part_count). S3 allows at most 10,000 parts of >= 5 MiB each."""
    part_size = max(part_mb * 2**20, 5 * 2**20, math.ceil(size_bytes / 10_000))
    return part_size, max(1, math.ceil(size_bytes / part_size))


def start_multipart(key: str, content_type: str) -> str:
    return s3.create_multipart_upload(Bucket=BUCKET, Key=key, ContentType=content_type)["UploadId"]


def presigned_part_urls(key: str, upload_id: str, part_count: int) -> list[dict]:
    return [
        {
            "part_number": n,
            "url": s3_public.generate_presigned_url(
                "upload_part",
                Params={"Bucket": BUCKET, "Key": key, "UploadId": upload_id, "PartNumber": n},
                ExpiresIn=config.PRESIGNED_URL_TTL_SECONDS,
            ),
        }
        for n in range(1, part_count + 1)
    ]


def complete_multipart(key: str, upload_id: str, parts: list[dict]) -> None:
    s3.complete_multipart_upload(
        Bucket=BUCKET,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [
                {"PartNumber": p["part_number"], "ETag": p["etag"]}
                for p in sorted(parts, key=lambda p: p["part_number"])
            ]
        },
    )


def abort_multipart(key: str, upload_id: str) -> None:
    s3.abort_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id)
