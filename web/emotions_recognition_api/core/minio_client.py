from pathlib import Path
from minio import Minio

ACCESS_KEY = "minio_user"
SECRET_KEY = "minio_password"
ENDPOINT = "0.0.0.0:9000"


class MinioClient:
    def __init__(self, bucket_name: str, endpoint=ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY) -> None:
        try:
            client = Minio(endpoint, access_key, secret_key, secure=False)
            self.client = client
            self.bucket_name = bucket_name
            if not client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as ex:
            print("Not able to connect minio / {}".format(ex))

    def upload_data(self, object_name: str, data, length):
        self.client.put_object(self.bucket_name, object_name, data, length)

    def download_file(self, object_name: str, file_path: Path):
        self.client.fget_object(self.bucket_name, object_name, str(file_path))

    def delete_file(self, object_name):
        self.client.remove_object(self.bucket_name, object_name)

