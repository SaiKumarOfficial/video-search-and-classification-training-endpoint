from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import  VideoFolderConfig
from src.entity.artifact_entity import ModelPusherArtifact,DataValidationArtifact
from src.constants.training_pipeline import SEQUENCE_LENGTH, IMAGE_HEIGHT,IMAGE_WIDTH
from src.constants.database import EMBEDDING_COLLECTION_NAME
from collections import namedtuple
import tensorflow as tf
from typing import List, Dict
import os,sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import json

VideoRecord = namedtuple("VideoRecord", ["video", "label", "s3_link"])


class VideoFolder(tf.keras.utils.Sequence):
    def __init__(self, label_map: Dict, data_validation_artifact: DataValidationArtifact):
        self.config = VideoFolderConfig(data_validation_artifact)  # Define your own VideoFolderConfig class
        # self.config.LABEL_MAP = label_map
        self.sequence_length = SEQUENCE_LENGTH
        self.video_records: List[VideoRecord] = []
        self.transform = self.transformations()

        file_list = os.listdir(self.config.ROOT_DIR)

        for class_path in file_list:
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
            videos = os.listdir(path)
            for video in tqdm(videos):
                video_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{video}""")
                self.video_records.append(VideoRecord(
                    video=video_path,
                    label=class_path,
                    s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path, video)
                ))


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
                normalized_frame = resized_frame/255
                # Add the frame to the list
                frames.append(normalized_frame)

            # Close the video file
            video_reader.release()

            # Convert the list of frames to a TensorFlow tensor
            # frames = tf.convert_to_tensor(frames, dtype=tf.float32)

            # Perform additional transformations as needed
            # For example, resizing, normalization, etc.

            return frames

        return preprocess_video

    def __len__(self):
        return len(self.video_records)

    def __getitem__(self, idx):
        record = self.video_records[idx]
        video, label, s3_link = record.video, record.label, record.s3_link

        # Example: Load video frames and apply transformations
        frames = self.transform(video)

        return frames, label, s3_link
    

class EmbeddingGenerator:
    def __init__(self, model_pusher_artifact: ModelPusherArtifact ):
        self.config = model_pusher_artifact # Assuming you have an EmbeddingsConfig class
        self.mongo = MongoDBClient()  # Assuming you have a MongoDBClient class
        
        # self.device = device
        self.embedding_model = self.load_model()
        self.embedding_model.trainable = False

    def load_model(self):
        model = tf.keras.models.load_model(self.config.saved_model_path)
        return tf.keras.Sequential(model.layers[:-1])

    def run_step(self, batch_size, video_frames, label, s3_link):
        records = dict()

        # Assuming images is a batch of images in TensorFlow format
        embeddings = self.embedding_model(video_frames)
        embeddings = embeddings.numpy()

        records['video_frames'] = embeddings.tolist()
        records['label'] = label.numpy().tolist()
        records['s3_link'] = s3_link

        df = pd.DataFrame(records)
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_record(records, collection_name= EMBEDDING_COLLECTION_NAME )

        return {"Response": f"Completed Embeddings Generation for {batch_size}."}


# if __name__ == "__main__":
    # # Step 1: Load label map and create VideoFolder instance
    # video_folder = VideoFolder(label_map={})  # Pass your label map here

    # # Step 2: Create DataLoader instance
    # dataloader = tf.data.Dataset.from_generator(
    #     video_folder.__iter__,
    #     output_signature=(
    #         tf.TensorSpec(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT , IMAGE_WIDTH, 3), dtype=tf.float32),
    #         tf.TensorSpec(shape=(), dtype=tf.int32),
    #         tf.TensorSpec(shape=(), dtype=tf.string),
    #     )
    # )
    # dataloader = dataloader.batch(64).shuffle(buffer_size=1000)

    # # Step 3: Create EmbeddingGenerator instance
    # embeds = EmbeddingGenerator()

#     # Step 4: Process each batch
#     for batch, values in tqdm(enumerate(dataloader)):
#         video_frames, target, link = values

#         # Step 5: Run EmbeddingGenerator for the current batch
#         result = embeds.run_step(batch, video_frames, target, link)

#         # Step 6: Print or handle the response
#         print(result)