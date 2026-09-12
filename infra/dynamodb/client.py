"""DynamoDB client factory.

Points at DynamoDB Local by default (.env DYNAMODB_ENDPOINT). To run against
real AWS DynamoDB, leave DYNAMODB_ENDPOINT empty and supply real credentials.
"""
from __future__ import annotations

import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


def _kwargs() -> dict:
    kwargs = {
        "region_name": os.getenv("DYNAMODB_REGION", "us-east-1"),
        "aws_access_key_id": os.getenv("DYNAMODB_AWS_ACCESS_KEY_ID", "dummy"),
        "aws_secret_access_key": os.getenv("DYNAMODB_AWS_SECRET_ACCESS_KEY", "dummy"),
        "config": Config(retries={"max_attempts": 5, "mode": "standard"}),
    }
    endpoint = os.getenv("DYNAMODB_ENDPOINT")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return kwargs


def table_name() -> str:
    return os.getenv("DYNAMODB_TABLE", "query_traces")


def dynamodb_client():
    return boto3.client("dynamodb", **_kwargs())


def dynamodb_resource():
    return boto3.resource("dynamodb", **_kwargs())