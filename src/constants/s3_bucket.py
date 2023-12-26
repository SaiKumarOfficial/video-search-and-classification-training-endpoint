import os                      
from from_root import from_root


ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_KEY"]
REGION_NAME = "ap-south-1"
BUCKET_NAME = "isro-documentary-videos"
KEY = "model"


ZIP_NAME = "artifacts.tar.gz"
# ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
#                           (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
#                           (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")]
