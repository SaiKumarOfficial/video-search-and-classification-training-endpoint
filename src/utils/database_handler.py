from src.constants import database
from pymongo import MongoClient
from typing import List, Dict, Any
import os

class MongoDBClient(object):
    def __init__(self):
        url = database.URL
        self.client = MongoClient(url)

    def insert_bulk_record(self, documents: List[Dict[str, Any]]):
        try:
            db = self.client[database.DATABASE_NAME]
            collection = database.COLLECTION_NAME
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            result = db[collection].insert_many(documents)
            return {"Response": "Success", "Inserted Documents": len(result.inserted_ids)}
        except Exception as e:
            raise e

    def get_collection_documents(self):
        try:
            db = self.client[database.DATABASE_NAME]
            collection = database.COLLECTION_NAME
            result = db[collection].find()
            return {"Response": "Success", "Info": result}
        except Exception as e:
            raise e

    def drop_collection(self):
        try:
            db = self.client[database.DATABASE_NAME]
            collection = database.COLLECTION_NAME
            db[collection].drop()
            return {"Response": "Success"}
        except Exception as e:
            raise e


if __name__ == "__main__":
    data = [
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"}
    ]

    mongo = MongoDBClient()
    print(mongo.insert_bulk_record(data))