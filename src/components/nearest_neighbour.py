from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import AnnoyConfig
from src.entity.artifact_entity import AnnoyArtifact
from src.constants.database import EMBEDDING_COLLECTION_NAME
from src.logger import logging
from annoy import AnnoyIndex
from typing_extensions import Literal
from tqdm import tqdm
import json
import os


class CustomAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        super().__init__(f, metric)
        self.label = []

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None:
        super().add_item(i, vector)
        self.label.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = 5, include_distances: bool = False):
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels = [self.label[link] for link in indexes]
        return labels

    def load(self, fn: str, prefault: bool = ...):
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...):
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Annoy(object):
    def __init__(self, annoy_config: AnnoyConfig):
        self.config =annoy_config
        self.mongo = MongoDBClient()
        self.result = self.mongo.get_collection_documents(collection_name= EMBEDDING_COLLECTION_NAME )

    def build_annoy_format(self):
        Ann = CustomAnnoy(32, 'euclidean')
        logging.info("Built Annoy format and adding data to it ")
        for i, record in tqdm(enumerate(self.result), total=8677):
            Ann.add_item(i, record["video_frames"], record["s3_link"])

        Ann.build(10)
        
        embeddings_dir = os.path.dirname(self.config.embeddings_store_path)
        os.makedirs(embeddings_dir, exist_ok= True)

        embeddings_store_file_path = self.config.embeddings_store_path
        logging.info("Saving the data in annoy")
        Ann.save(embeddings_store_file_path)

        annoy_artifact = AnnoyArtifact(embeddings_store_file_path=  embeddings_store_file_path)

        return annoy_artifact

    def run_step(self):
        annoy_artifact= self.build_annoy_format()
        return annoy_artifact


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()
