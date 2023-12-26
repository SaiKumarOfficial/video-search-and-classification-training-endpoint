# Video Classification Training Endpoint

## Classification Model

### LRCN (Long-Short Term Recurrent Convolutional Network)

![lrcn_model](https://github.com/SaiKumarOfficial/video-streaming-training-endpoint/assets/95096218/9aaed427-a730-40e0-ab38-f8b311f405e8)

We have employed the LRCN classification model for video categorization. LRCN combines the power of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), making it effective for sequences of data like video frames. This model was chosen due to its favorable performance on our unique dataset, which comprises ISRO documentary video data. Training from scratch proved essential for our dataset, and LRCN demonstrated superior results with advantages such as reduced memory storage requirements and faster training times.You can look at this reasearch paper at [here](https://arxiv.org/pdf/1411.4389.pdf).

## Vector Search

For efficient video search functionality, we store video embeddings in MongoDB, obtained by excluding the last layer of the LRCN model. To facilitate quick and approximate nearest-neighbor searches, we leverage Spotify's ANNOY (Approximate Nearest Neighbour Oh Yeah) algorithm.

## MLops Workflow

<!-- We adopt Google's Level 1 MLOps architecture for streamlined and efficient workflow management. The architecture encompasses various stages, including data preparation, model training, deployment, and monitoring. -->

![TrainingPipelineWorkflow](https://github.com/SaiKumarOfficial/video-streaming-training-endpoint/assets/95096218/d0c8be26-f934-426b-9f27-b234f0276c51)

## Tech Stacks

1. **Python**
2. **MongoDB**
3. **Amazon S3**
4. **GitHub Actions**

## Setup Steps

1. Create a new conda environment:
    ```bash
    conda create --name your_env_name python=3.9
    ```

2. Activate the environment:
    ```bash
    conda activate your_env_name
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:

    - MONGODB_USERNAME
    - MONGODB_PASSWORD
    - MONGODB_URL
    - ACCESS_KEY_ID
    - AWS_SECRET_KEY
    - S3_BUCKET_NAME

5. Run the pipeline:
    ```bash
    python src/pipeline/pipeline.py
    ```

Feel free to reach out for any additional support or inquiries!