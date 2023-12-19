from src.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.storage_handler import S3Connector
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
import os,sys
import random,shutil

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            # self.training_pipeline_config = TrainingPipelineConfig()
            # self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def download_data_fromS3(self):
        try:
            logging.info(" Fetching Data from S3 ")

            raw_data_file_path = self.data_ingestion_config.raw_data_dir
            
            dir_path = os.path.dirname(raw_data_file_path)
            os.makedirs(dir_path,exist_ok= True)

            os.system(f"aws s3 sync s3://isro-documentary-videos/videos/ {raw_data_file_path} --no-progress")

            logging.info(" Fetching Completed ")
            return raw_data_file_path
        except Exception as e:
            raise CustomException(e,sys)
    def movefiles(self,videos,label_folder, label_output):
        try:
            for video in videos:
                src_path = os.path.join(label_folder, video)
                dest_path = os.path.join(label_output, video)
                shutil.move(src_path, dest_path)
        except Exception as e:
            raise CustomException(e,sys)


    def split_videos(self,input_folder, output_train_folder, output_test_folder, split_ratio=0.8, seed=None):
        try:
            if seed is not None:
                random.seed(seed)
            logging.info("Split the Videos into train and test folders")
            for label in os.listdir(input_folder):
                label_folder = os.path.join(input_folder, label)
                
                # Ensure it's a directory
                if os.path.isdir(label_folder):
                    video_files = [f for f in os.listdir(label_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
                    random.shuffle(video_files)

                    split_index = int(len(video_files) * split_ratio)
                    train_videos = video_files[:split_index]
                    test_videos = video_files[split_index:]

                    # Create output directories if they don't exist
                    train_label_output = os.path.join(output_train_folder, label)
                    test_label_output = os.path.join(output_test_folder, label)
                    os.makedirs(train_label_output, exist_ok=True)
                    os.makedirs(test_label_output, exist_ok=True)

                    # Move files to the corresponding train and test folders
                    self.movefiles(train_videos,label_folder,train_label_output)
                    self.movefiles(test_videos,label_folder,test_label_output)

            logging.info(f"Successfully splitted raw data into train and test with {split_ratio} ratio")
            return True
        except Exception as e:
                raise CustomException(e,sys)
    def run_step(self):
        try:
            logging.info("==================Start the data ingestion phase==============")

            input_folder = self.download_data_fromS3()
            # output_train_folder = self.data_ingestion_config.training_data_dir
            # output_test_folder = self.data_ingestion_config.testing_data_dir
            # split_ratio = self.data_ingestion_config.train_test_split_ratio
            # seed_value = self.data_ingestion_config.seed_value
            # self.split_videos(input_folder, output_train_folder, output_test_folder, 
            #             split_ratio = split_ratio,
            #             seed=seed_value)
            data_ingestion_artifact = DataIngestionArtifact(raw_data_dir= input_folder)
            logging.info("=================== Successfully Completed data ingestion ==================")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    
if __name__ == "__main__":
    
    data_ingestion = DataIngestion(data_ingestion_config= DataIngestionConfig)
    
    print(data_ingestion.run_step())