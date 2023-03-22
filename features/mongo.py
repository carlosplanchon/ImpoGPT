import os

from pymongo import MongoClient


class MongoQuerier:
    def __init__(self):
        self.uri = os.environ.get("MONGO_URI")
        self.client = MongoClient(self.uri)
        self.db = self.client["test"]
        self.prompts_collection = self.db.get_collection("queries")

    def insert_documents(self, docs, hashed_api_key: str):
        for doc in docs:
            doc["hash_id"] = hashed_api_key
        self.prompts_collection.insert_many(docs)

    def search_docs(self, tag: str):
        return self.prompts_collection.find({'tag': tag})

    def get_all(self):
        return self.prompts_collection.find({})

    def get_by_hash(self, hash_id: str, include_process: bool = False):
        mongo_filter = {'hash_id': hash_id}
        if not include_process:
            mongo_filter['tag'] = 'final_answer'
        return self.prompts_collection.find(mongo_filter)

