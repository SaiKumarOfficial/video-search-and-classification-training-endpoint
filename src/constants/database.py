import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_NAME = 'Isro-project'
LABEL_COLLECTION_NAME = 'labels-data'
# COLLECTION_NAME =  "embeddings"

# COLLECTION_NAME = 'videos-data'
EMBEDDING_COLLECTION_NAME = "Embeddings"

USERNAME = os.environ['DATABASE_USERNAME']
PASSWORD  = os.environ['DATABASE_PASSWORD']
URL  = os.environ['DATABASE_URL']
