from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.utils.database_handler import MongoDBClient
from src.constants.training_pipeline import SCHEMA_KEY,VALID_EXTENSIONS
from src.utils.common import write_yaml, read_yaml
from src.exception import CustomException
from src.logger import logging
import os,sys,yaml
import shutil


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self.client = MongoDBClient()

        except Exception as e:
            raise CustomException(e,sys)
        
    def create_labels_schema(self):
        try:
        # Transform data into the desired format
            logging.info("Creating labels schema...")
            labels = []
            result = self.client.get_collection_documents()
            data = list(result)
            for key,value in data[0].items():
                if key=='_id':
                    continue
                labels.append(value)
            schema_file_path  = self.data_validation_config.labels_schema_file_path
            # Create YAML file
    
            write_yaml(schema_file_path,labels)
            logging.info("Return the yaml file which contains all the labels")
            return schema_file_path
        except Exception as e:
            raise CustomException(e,sys)
    def valid_labels(self,data_dir, schema_path):
        try:
            logging.info("Validating the labels from mongobd and from s3 ")
            data = read_yaml(file_path=schema_path)
            original_labels = data[SCHEMA_KEY]
            stored_labels = os.listdir(data_dir)
            original_length, stored_length = len(original_labels) ,len(stored_labels)

            if original_length != stored_length:
                if original_length > stored_length: 

                    missing_from_stored_labels = [element for element in original_labels if element not in stored_labels]
                    logging.info(f"There are {missing_from_stored_labels} Labels are missing from stored labels")
                else:
                    missing_from_original_labels = [element for element in stored_labels if element not in original_labels]

                    logging.info(f"There are {missing_from_original_labels} Labels are missing from original labels")
                raise CustomException("Labels are not equal",sys)
            
            logging.info(f"There are {len(stored_labels)} valid no.of Labels are present")
        except Exception as e:
            raise CustomException(e,sys)

    def count_videos(self,directory, allowed_extensions):

        logging.info("Count the no.of videos from each directory")
        # Iterate through each directory
        count_list = []
        for label_directory in os.listdir(directory):
            label_path = os.path.join(directory, label_directory)

            # Skip if it's not a directory
            if not os.path.isdir(label_path):
                continue
            # Count videos in the current directory
            videos = [file for file in os.listdir(label_path) if file.endswith(tuple(allowed_extensions))]
            video_count = len(videos)
            count_list.append(video_count)
        return count_list
    
    def valid_no_of_videos(self,src_directory, valid_directory,min_videos,allowed_extensions):
        try:

            videos_count = self.count_videos(src_directory,allowed_extensions)
            logging.info("Validating no.of videos and consider minmum no.of videos from each directory")

            min_video_count = min(videos_count)
            if min_video_count < min_videos:
                logging.info(f"There are only minimum of {min_video_count} videos, but we should require {min_videos}")
                raise CustomException(f"{min_videos-min_video_count} Videos are required in each directory!!",sys)
            min_videos = min(min_video_count,min_videos)

            if not os.path.exists(valid_directory):
                os.makedirs(valid_directory)

            # Iterate through each directory
            for label_directory in os.listdir(src_directory):
                label_path = os.path.join(src_directory, label_directory)
                valid_path = os.path.join(valid_directory, label_directory)
                if not os.path.exists(valid_path):
                    os.makedirs(valid_path)
                # Skip if it's not a directory
                if not os.path.isdir(label_path):
                    continue

                # Validate and copy videos
                videos = [file for file in os.listdir(label_path) if file.endswith(tuple(allowed_extensions))]
                if len(videos) >= min_videos:
                    # Copy the minimum required videos to the valid directory
                    for video in videos[:min_videos]:
                        src_path = os.path.join(label_path, video)
                        dest_path = os.path.join(valid_path, video)
                        shutil.copy2(src_path, dest_path)
                        logging.info(f"Copied: {src_path} to {dest_path}")
            logging.info("Returning the valid data directory")
            return valid_directory
        except Exception as e:
            raise CustomException(e,sys)

    def run_steps(self):
        logging.info("===========  Starting Data Validation Phase =============")
        
        schema_file_path= self.create_labels_schema()
        raw_data_dir = self.data_ingestion_artifact.raw_data_dir
        valid_directory_path = self.valid_no_of_videos(
            src_directory= raw_data_dir,
            valid_directory= self.data_validation_config.valid_data_dir,
            min_videos= self.data_validation_config.minimum_required_videos,
            allowed_extensions = VALID_EXTENSIONS
        )
        data_validation_artifact = DataValidationArtifact(
                                            labels_schema_file_path=schema_file_path,
                                             valid_data_dir= valid_directory_path )
        logging.info("=========== Successfully completed Data Validation Phase =============")
        return data_validation_artifact

        