from src.logger import logging
from src.constants.training_pipeline import SCHEMA_KEY
import numpy as np           
import time
import yaml,os

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


# print(get_unique_filename("model", "pth"))  # model_name ,extension
