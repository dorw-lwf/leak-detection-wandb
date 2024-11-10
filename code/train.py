import wandb
import boto3
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk import launch

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/depend/code/requirements.txt",
])

import argparse
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def get_train_data(train_dir):
    #train_data = pd.read_csv(os.path.join(train_dir, "train.csv"))
    
    
    print("train_dir",train_dir)
    #x_train = train_data[feature_columns].to_numpy()
    x_train = np.load(train_dir+"/X_train.npy")
    y_train = np.load(train_dir+"/y_train.npy")
    #y_train = train_data[label_column].to_numpy()
    print("x train", x_train.shape, "y train", y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):
    
    print("train_dir",test_dir)
    x_test = np.load(test_dir+"/X_test.npy")
    y_test = np.load(test_dir+"/y_test.npy")
    print("x test", x_test.shape, "y test", y_test.shape)
    return x_test, y_test


def get_model(num_classes):
    
    
    input_shape = (5,29,89,1)
    
    model = Sequential(
        [
            Input(shape=input_shape),
            layers.Conv3D(128,(3,3,3),activation='relu',input_shape=(5,29,89,1),bias_initializer=Constant(0.01)),
            layers.Conv3D(128,(3,3,3),activation='relu',bias_initializer=Constant(0.01)),
            layers.MaxPooling3D((2,2,2), padding='same'),
            layers.Conv3D(64,(3,3,3),activation='relu', padding='same'),
            layers.Conv3D(64,(3,3,3),activation='relu', padding='same'),
            layers.MaxPooling3D((2,2,2), padding='same'),
            layers.Dropout(0),
            layers.Flatten(),
            layers.Dropout(0),
            layers.Dense(64, activation="softmax"),
            layers.Dropout(0),
            layers.Dense(32, activation="softmax"),
            layers.Dropout(0),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    #inputs = tf.keras.Input(shape=(8,))
    #hidden_1 = tf.keras.layers.Dense(8, activation="tanh")(inputs)
   # hidden_2 = tf.keras.layers.Dense(4, activation="sigmoid")(hidden_1)
   # outputs = tf.keras.layers.Dense(1)(hidden_2)
   # return tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    args, _ = parse_args()

    print("Training data location: {}".format(args.train))
    print("Test data location: {}".format(args.test))
    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print(
        "batch_size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, learning_rate)
    )
    num_classes = 2

    model = get_model(num_classes)
    
    batch_size = args.batch_size
    epochs = args.epochs
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    
    #optimizer = tf.keras.optimizers.SGD(learning_rate)
    #model.compile(optimizer=optimizer, loss="mse")
    
    region = "eu-central-1"
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    hot_y_train = np_utils.to_categorical(encoded_Y)
    
    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_Y = encoder.transform(y_test)
    hot_y_test = np_utils.to_categorical(encoded_Y)
    
    
    model.fit(
        x_train, hot_y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, hot_y_test)
    )

    # evaluate on test set
    scores = model.evaluate(x_test, hot_y_test, batch_size, verbose=2)
    print("Loss, Accuracy :", scores)

    # save model
    model.save(args.sm_model_dir + "/1.keras")
