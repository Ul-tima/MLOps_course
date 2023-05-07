import io
from unittest import mock

import pytest
from core.minio_client import MinioClient


@pytest.fixture()
def bucket_name():
    return "test-bucket"


@pytest.fixture()
def object_name():
    return "test"


@pytest.fixture()
def stream():
    return io.BytesIO(b"abc")


@pytest.fixture()
def length():
    return 3


@pytest.fixture()
def file_path():
    return "file_path"


@pytest.fixture()
def minio(bucket_name: str):
    return MinioClient(bucket_name)


def test_upload_data(bucket_name: str, stream: io.BytesIO, length: int, minio: MinioClient):
    # Act
    with mock.patch.object(minio.client, "put_object") as patched:
        minio.upload_data("test", stream, length)

    # Assert
    patched.assert_called_once_with(bucket_name, "test", stream, length)


def test_download_file(bucket_name: str, object_name: str, file_path: str, minio: MinioClient):
    # Act
    with mock.patch.object(minio.client, "fget_object") as patched:
        minio.download_file(object_name, file_path)

    # Assert
    patched.assert_called_once_with(bucket_name, object_name, file_path)


def test_delete_file(bucket_name: str, object_name: str, minio: MinioClient):
    # Act
    with mock.patch.object(minio.client, "remove_object") as patched:
        minio.delete_file(object_name)

    # Assert
    patched.assert_called_once_with(bucket_name, object_name)
