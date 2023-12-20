from src.entity.config_entity import DataPreparationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataPreparationArtifact
from src.constants.training_pipeline import IMAGE_HEIGHT,IMAGE_WIDTH, SEQUENCE_LENGTH, SCHEMA_KEY
from src.utils.common import read_yaml
from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os,sys
import numpy as np 


class DataPreparation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                    data_preparation_config: DataPreparationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_preparation_config = data_preparation_config
        except Exception as e:
            raise CustomException(e,sys)

    def resize_frame(self,image):
        return cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
    def frames_extraction(self, video_path):
        
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)

        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

        for frame_counter in range(SEQUENCE_LENGTH):

            # Set the current frame position of the video
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # reading the frame from video
            success , frame = video_reader.read()
            if not success:
                break

            resized_frame = self.resize_frame(frame)
            normalized_frame = resized_frame/255

            # Append the normalized frame into the frames list
            frames_list.append(normalized_frame)
        video_reader.release()
        return frames_list
    
    def create_dataset(self,schema_file_path,data_dir):
        logging.info("Started creating dataset and return features, labels and corresponding video file paths")
        features = []
        labels = []
        video_files_path = []
        data = read_yaml(schema_file_path)
        labels_list = data[SCHEMA_KEY]
        # Iterating through all the classes mentioned in the classes list
        for class_index ,class_name in enumerate(labels_list):

            # display the name of the class
            logging.info(f"Extracting Data of class: {class_name}")
            files_list = os.listdir(os.path.join(data_dir,class_name))

            for  file_name in files_list:

                # get the complete video path
                video_file_path = os.path.join(data_dir, class_name , file_name)

                # Extract the frames of the video file
                frames = self.frames_extraction(video_file_path)
                if len(frames) == SEQUENCE_LENGTH:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_path.append(video_file_path)

        # Convertion the list to numpy arrays
        features = np.asarray(features)
        labels = np.array(labels)

        # Return the frames ,class index , video file path

        return features ,labels , video_files_path


    def perform_encoding(self, labels):
        logging.info("Performing One hot encoding ")
        one_hot_encoded_labels = to_categorical(labels)
        return one_hot_encoded_labels

    def save_file(self,path, data):
        directory, filename = os.path.split(path)
    
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        resultant_path = os.path.join(directory, filename)
        np.save(resultant_path, data)

        return resultant_path

    def run_steps(self):
        logging.info("============= Starting data preparation phase =================")
        valid_data_dir = self.data_validation_artifact.valid_data_dir
        schema_file_path = self.data_validation_artifact.labels_schema_file_path

        
        features, labels, video_files_path =self.create_dataset( schema_file_path= schema_file_path,
                            data_dir= valid_data_dir)
        encoded_labels = self.perform_encoding(labels)

        logging.info(f"Splitting the data into train and test in {self.data_preparation_config.train_test_split_ratio} test ratio")
        features_train, features_test, labels_train, labels_test = train_test_split(features, encoded_labels,
                                                             test_size=self.data_preparation_config.train_test_split_ratio, 
                                                             shuffle= self.data_preparation_config.shuffle,
                                                             random_state = self.data_preparation_config.seed_value)
        logging.info("Save training data")
        features_train_file_path = self.save_file(self.data_preparation_config.features_train_file_path, features_train)
        features_test_file_path = self.save_file(self.data_preparation_config.features_test_file_path, features_test)
        labels_train_file_path  = self.save_file(self.data_preparation_config.labels_train_file_path, labels_train)
        labels_test_file_path = self.save_file(self.data_preparation_config.labels_test_file_path , labels_test)

        logging.info("=================== Successfully completed data preparation phase ==============")
        return DataPreparationArtifact(
                        features_train_file_path= features_train_file_path,
                        labels_train_file_path= labels_train_file_path,
                        features_test_file_path= features_test_file_path,
                        labels_test_file_path= labels_test_file_path
        )   
        
