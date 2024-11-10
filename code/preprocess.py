import sys
import subprocess
import os
import wandb
import boto3
from wandb.integration.keras import WandbMetricsLogger
from wandb.sdk import launch

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/depend/code/requirements.txt",
])
from my_package.read_DAS_hdf5 import load_single_DAS_file, list_hdf5_files_in_dir, load_multi_DAS_file, generate_training_set, generate_training_set_spectrogram, moving_average, define_butterworth_highpass, filtering, spectrogram

#from my_package.DAS_filtering import moving_average,define_butterworth_highpass,filtering,spectrogram
import s3fs
import h5py
import glob
import numpy as np
import pandas as pd
import os
import json
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tarfile
from sklearn.model_selection import train_test_split

try:
    from sagemaker_containers.beta.framework import (
        content_types,
        encoders,
        env,
        modules,
        transformer,
        worker,
        server,
    )
except ImportError:
    pass


base_dir = "/opt/ml/processing"
base_output_dir = "/opt/ml/output/"

if __name__ == "__main__":
    
    data_location_str = base_dir+"/input/"

    print("data_location_str", data_location_str)
    lst = os.listdir(data_location_str) # your directory path
    number_files = len(lst)
    print(number_files)
    
    channels = 376
    file_names = list_hdf5_files_in_dir(data_location_str)
    
    print(file_names)
    
    DAS_fitlered_data = load_multi_DAS_file(file_names, channels)
    
    print("DAS data shape", DAS_fitlered_data.shape)
    
    
    ## 100 lm/m sample 1 ( 60 2 )
    channel_start_leak = 188
    channel_end_leak = 199
    capture_period_leak = [300000, 2182000]
    fs = 20000
    training_data_leak = generate_training_set_spectrogram(DAS_fitlered_data, channel_start_leak, channel_end_leak,capture_period_leak,fs)
    training_label_leak = np.ones(training_data_leak.shape[0])

    ## no leak
    channel_start_no_leak = 148
    channel_end_no_leak = 270
    capture_period_no_leak = [0, 200000]
    training_data_noleak = generate_training_set_spectrogram(DAS_fitlered_data, channel_start_no_leak, channel_end_no_leak,capture_period_no_leak,fs)
    training_label_noleak = np.zeros(training_data_noleak.shape[0])

    training_data = np.concatenate([training_data_leak,training_data_noleak ])
    training_label = np.concatenate([training_label_leak,training_label_noleak ])

    train = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2],training_data.shape[3], 1)
    
    seed = 7
    X_train, X_test, y_train, y_test = train_test_split(train, training_label, test_size=0.33, random_state=seed)

    
    np.save(f"{base_dir}/train/X_train.npy",X_train)
    np.save(f"{base_dir}/train/y_train.npy",y_train)

    np.save(f"{base_dir}/test/X_test.npy",X_test)
    np.save(f"{base_dir}/test/y_test.npy",y_test)
    
    #train_dataset.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    #test_dataset.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    #joblib.dump(scaler, "model.joblib")
    #with tarfile.open(f"{base_dir}/scaler_model/model.tar.gz", "w:gz") as tar_handle:
        #tar_handle.add(f"model.joblib")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(feature_columns) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns + [label_column]
        elif len(df.columns) == len(feature_columns):
            # This is an unlabelled example.
            df.columns = feature_columns

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append(row)
        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
