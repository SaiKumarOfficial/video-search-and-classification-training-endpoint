import os

DATABASE_NAME = 'Isro-project'
COLLECTION_NAME = 'videos-data'
COLLECTION= "Embeddings"

USERNAME = os.environ['DATABASE_USERNAME']
PASSWORD  = os.environ['DATABASE_PASSWORD']
URL  = f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.edjcajk.mongodb.net"
