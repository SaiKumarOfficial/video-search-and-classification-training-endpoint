from src.logger import logging
from src.constants.training_pipeline import SCHEMA_KEY, MODEL_FILE_NAME, SAVED_MODEL_DIR
from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception import CustomException
import numpy as np           
import time
import yaml,os,sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
import tensorflow as tf

def set_seed(seed_value: int = 42) -> None:
    np.random.seed(seed_value)


def get_unique_filename(filename, ext):
    return time.strftime(f"{filename}_%Y_%m_%d_%H_%M.{ext}")

def read_yaml(file_path):

    try:
        with open(file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
    
def write_yaml(file_path, data):
    
    try:
        # Create parent directory if it doesn't exist

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as yaml_file:
                yaml.dump({SCHEMA_KEY: data}, yaml_file, default_flow_style=False)
        logging.info(f"Labels schema file  has been created at {file_path}.")
        return True
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        return False
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def load_model(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        model = tf.keras.models.load_model(file_path)
        return model
    except Exception as e:
        raise CustomException(e, sys) from e

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        conf_metrics = confusion_matrix(y_true, y_pred)
        model_accuracy =  accuracy_score(y_true, y_pred)
        model_f1_score = f1_score(y_true,y_pred,average='weighted')
        model_recall_score = recall_score(y_true,y_pred,average='weighted')
        model_precision_score = precision_score(y_true,y_pred,average='weighted')
        classification_metric =  ClassificationMetricArtifact(f1_score= model_f1_score,
                precision_score= model_precision_score,
                recall_score= model_recall_score,
                accuracy= model_accuracy,
                conf_matrix = conf_metrics
                )

        return classification_metric
    except Exception as e:
        raise CustomException(e,sys)

class ModelResolver:
    # it is helping us to findout the best model
    def __init__(self,model_dir= SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise e
    def get_best_model_path(self) -> str:
        try:
            timestamp = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamp)
            latest_model_path = os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e
        
    def is_model_exist(self)->str:
        try:
            if not os.path.exists(self.model_dir):
                return False
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False
            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False
            return True
        except Exception as e:
            raise e
        

# print(get_unique_filename("model", "pth"))  # model_name ,extension
