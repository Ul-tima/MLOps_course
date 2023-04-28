import io

import pytest
from unittest import mock
from core.minio_client import MinioClient


def test_upload_data():
    # Arrange
    bucket_name = 'test-bucket'
    stream = io.BytesIO(b"abc")
    length = 3
    minio = MinioClient(bucket_name)

    # Act
    with mock.patch.object(minio.client, "put_object") as patched:
        minio.upload_data('test', stream, length)

    # Assert
    patched.assert_called_once_with(bucket_name, 'test', stream, length)


def test_download_file():
    # Arrange
    bucket_name = 'test-bucket'
    object_name = 'test'
    file_path = 'test-path'
    minio = MinioClient(bucket_name)

    # Act
    with mock.patch.object(minio.client, "fget_object") as patched:
        minio.download_file(object_name, file_path)

    # Assert
    patched.assert_called_once_with(bucket_name, object_name, file_path)


def test_delete_file():
    # Arrange
    bucket_name = 'test-bucket'
    object_name = 'test'
    minio = MinioClient(bucket_name)

    # Act
    with mock.patch.object(minio.client, "remove_object") as patched:
        minio.delete_file(object_name)

    # Assert
    patched.assert_called_once_with(bucket_name, object_name)
