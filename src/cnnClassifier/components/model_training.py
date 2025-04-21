import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
import numpy as np
import math
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        self.model.summary() 

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=preprocess_input,  #Fixed: use VGG16 preprocessing
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def calculate_class_weights(self):
        labels = self.train_generator.classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(class_weights))

    def fine_tune_model(self):
        for layer in self.model.layers[:-4]:
            layer.trainable = False

        for layer in self.model.layers[-4:]:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
    )


    def train(self):
        self.steps_per_epoch = math.ceil(self.train_generator.samples / self.train_generator.batch_size)
        self.validation_steps = math.ceil(self.valid_generator.samples / self.valid_generator.batch_size)

        class_weights = self.calculate_class_weights()  #Handle imbalance

        # Optional: Fine-tune if you've already trained the top layers
        if self.config.params_fine_tune:  #Add this to your config if needed
            self.fine_tune_model()

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            class_weight=class_weights  #Add class weights
        )

        self.save_model(
            path=self.config.training_model_path,
            model=self.model
        )
