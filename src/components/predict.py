from src.components.nearest_neighbour import CustomAnnoy
from src.constants.training_pipeline import EMBEDDINGS_LENGTH,SEARCH_MATRIX, SEQUENCE_LENGTH, IMAGE_HEIGHT,IMAGE_WIDTH,NUMBER_OF_PREDICTIONS
from src.entity.artifact_entity import AnnoyArtifact, ModelPusherArtifact

import tensorflow as tf
import numpy as np
import cv2

class Prediction(object):

    def __init__(self, model_pusher_artifact: ModelPusherArtifact,annoy_artifact: AnnoyArtifact):
        self.device = 'cpu'
        self.model_pusher_artifact = model_pusher_artifact
        self.sequence_length = SEQUENCE_LENGTH
        self.annoy_artifact = annoy_artifact
        self.ann = CustomAnnoy(EMBEDDINGS_LENGTH, SEARCH_MATRIX)
        self.ann.load(self.annoy_artifact.embeddings_store_file_path)
        self.estimator = self.load_model()
        self.transforms = self.transformations()

    def load_model(self):
        model = tf.keras.models.load_model(self.model_pusher_artifact.saved_model_path)
        return tf.keras.Sequential(model.layers[:-1])
    
    
    def transformations(self):

        def preprocess_video(video_path):
        # Open the video file
            video_reader = cv2.VideoCapture(str(video_path))

            # Get the total number of frames in the video
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the skip frames window based on the sequence length
            skip_frames_window = max(int(video_frames_count / self.sequence_length), 1)

            # Initialize an empty list to store video frames
            frames = []

            # Loop through frames based on the sequence length
            for frame_counter in range(self.sequence_length):
                # Set the current frame position of the video
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                # Read the frame
                ret, frame = video_reader.read()

                # Check if the frame is successfully read
                if not ret:
                    break

                # Convert the frame to RGB format (assuming BGR format from OpenCV)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized_frame = resized_frame/255.0
                # Add the frame to the list
                frames.append(normalized_frame)
            frames = np.array(frames)
            # Close the video file
            video_reader.release()

            return frames

        return preprocess_video

    def generate_embeddings(self, video_frames):
        
        video_frame = self.estimator.predict(video_frames)
        return video_frame
    
    def generate_links(self, embedding):
        return self.ann.get_nns_by_vector(embedding, NUMBER_OF_PREDICTIONS)
    
    def run_predictions(self, video_file_path):
        video_frames = self.transforms(video_file_path)

        video_frames = video_frames.reshape((1,) + video_frames.shape)
        embedding = self.generate_embeddings(video_frames)
        print(embedding)
        return self.generate_links(embedding[0])
    