from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataPreparationArtifact,DataValidationArtifact
from src.exception import CustomException
from src.constants import training_pipeline 
from src.logger import logging
from src.utils.common import get_classification_metrics,read_yaml,load_numpy_array_data
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import TimeDistributed, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import datetime as dt
import  os, sys
import numpy as np
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                    data_validation_artifact: DataValidationArtifact,
                    data_preparation_artifact: DataPreparationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_preparation_artifact = data_preparation_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
    
    def create_LRCN_model(self, schema_path ):
        try:
            data = read_yaml(file_path=schema_path)
            labels = data[training_pipeline.SCHEMA_KEY]

            logging.info(" Constructing  model architecture...")
            model = Sequential()

            model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation ='relu'),
                                    input_shape = (training_pipeline.SEQUENCE_LENGTH, training_pipeline.IMAGE_HEIGHT, training_pipeline.IMAGE_WIDTH, 3)))
            model.add(TimeDistributed(MaxPooling2D((4, 4))))
            # model.add(TimeDistributed(Dropout(0.25)))

            model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))

            model.add(TimeDistributed(MaxPooling2D((4, 4))))
            model.add(TimeDistributed(Dropout(0.2)))

            model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))

            model.add(TimeDistributed(MaxPooling2D((2, 2))))
            model.add(TimeDistributed(Dropout(0.2)))

            model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))

            model.add(TimeDistributed(MaxPooling2D((2, 2))))
            # model.add(TimeDistributed(Dropout(0.25)))

            model.add(TimeDistributed(Flatten()))

            model.add(LSTM(32))

            model.add(Dense(len(labels), activation = 'softmax'))
    
            # Display the models summary.
            # model.summary()

            # Return the constructed LRCN model.
            return model
    
        except Exception as e:
            raise CustomException(e,sys)
        
    def compile_model(self, model):
        try:
            logging.info("Compiling the model... ")
            optimizer = Adam(learning_rate= training_pipeline.OPTIMIZER_LR)
            model.compile(loss= training_pipeline.LOSS, optimizer=optimizer, metrics= training_pipeline.METRICS)

            return True
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_model(self, model, features_train,labels_train, 
                        features_test, labels_test ):
        try:
            logging.info("Start training the model")
            early_stopping = EarlyStopping(patience = training_pipeline.ER_STOP_PATIENCE, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)
            model_training_history = model.fit(features_train,labels_train, batch_size=training_pipeline.BATCH_SIZE,epochs = training_pipeline.EPOCHS,
                                    shuffle = True, callbacks = [early_stopping, reduce_lr] ,validation_data = (features_test, labels_test))
            logging.info("Successfully Completed model training")

            return model_training_history
        except Exception as e:
            raise CustomException(e,sys)

    def evaluate_model(self,y_predict, y_true):
        # Convert predictions to class labels
        predicted_labels = np.argmax(y_predict, axis=1)

        # Convert true labels to class labels
        true_labels = np.argmax(y_true, axis=1)
        
        # Calculate accuracy
        classification_metrics = get_classification_metrics(true_labels,predicted_labels)

        return classification_metrics
        

    def save_model(self,model_file_path, model):
        try:

            model.save(model_file_path)

            return True
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_steps(self):
        try:
            schema_file_path = self.data_validation_artifact.labels_schema_file_path
            model = self.create_LRCN_model(schema_file_path)

            self.compile_model(model= model)

            features_train = load_numpy_array_data(self.data_preparation_artifact.features_train_file_path)
            labels_train = load_numpy_array_data(self.data_preparation_artifact.labels_train_file_path)
            features_test = load_numpy_array_data(self.data_preparation_artifact.features_test_file_path)
            labels_test = load_numpy_array_data(self.data_preparation_artifact.labels_test_file_path)

            self.train_model( model = model, features_train = features_train, labels_train=labels_train, 
                                features_test= features_test, labels_test= labels_test)
            # Evaluating train data
            labels_train_pred = model.predict(features_train)
            classification_train_metrics = self.evaluate_model(y_predict=labels_train_pred,y_true= labels_train)

            # Evaluating test data
            labels_test_pred = model.predict(features_test)
            classification_test_metrics = self.evaluate_model(y_predict=labels_test_pred,y_true= labels_test)
            diff =  abs(classification_train_metrics.accuracy - classification_test_metrics.accuracy)
            
            # if diff > self.model_trainer_config.overfitting_underfitting_threshold:
            #     raise Exception("Model is not good , try to do more experimentations!!")

            logging.info("Save the model...")
            model_dir_path = os.path.dirname(self.model_trainer_config.model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            model_file_path = os.path.join(self.model_trainer_config.model_file_path)
            self.save_model(model_file_path, model)
            
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path= model_file_path,
                                                        train_metric_artifact= classification_train_metrics,
                                                        test_metric_artifact= classification_test_metrics)

            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e,sys)

