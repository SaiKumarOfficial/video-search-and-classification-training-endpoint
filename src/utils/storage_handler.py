from src.constants import s3_bucket
import tarfile
from boto3 import Session
import os
from src.entity.artifact_entity import AnnoyArtifact, ModelPusherArtifact
from src.logger import logging


class S3Connector(object):
    def __init__(self, annoy_artifact: AnnoyArtifact, model_pusher_artifact: ModelPusherArtifact):
        self.config = s3_bucket
        self.session = Session(aws_access_key_id=self.config.ACCESS_KEY_ID,
                               aws_secret_access_key=self.config.SECRET_KEY,
                               region_name=self.config.REGION_NAME)
        self.client = self.session.client("s3")
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(self.config.BUCKET_NAME)
        self.annoy_artifact = annoy_artifact
        self.model_pusher_artifact = model_pusher_artifact
        self.ZIP_PATHS = [ (os.path.join(os.path.dirname(self.annoy_artifact.embeddings_store_file_path), 'embeddings.json'),'embeddings.json'),
                          (self.annoy_artifact.embeddings_store_file_path,'embeddings.ann'), (self.model_pusher_artifact.saved_model_path, 'model.h5')]

    def zip_files(self):
        folder = tarfile.open(self.config.ZIP_NAME, "w:gz")
        print(folder)
        for path, name in self.ZIP_PATHS:
            folder.add(path, name)
        folder.close()

        logging.info("Upload artifacts into S3")
        self.s3.meta.client.upload_file(self.config.ZIP_NAME, self.config.BUCKET_NAME,
                                        f'{self.config.KEY}/{self.config.ZIP_NAME}')
        os.remove(self.config.ZIP_NAME)
        return {"Response": "Successfully uploaded"}

    def pull_artifacts(self):
        self.bucket.download_file(f'{self.config.KEY}/{self.config.ZIP_NAME}', self.config.ZIP_NAME)
        folder = tarfile.open(self.config.ZIP_NAME)
        folder.extractall()
        folder.close()
        os.remove(self.config.ZIP_NAME)



if __name__ == "__main__":
    connection = S3Connector()
    # connection.zip_files()
    # connection.pull_artifacts()
