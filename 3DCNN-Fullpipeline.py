# run jupytext --to notebook 3DCNN-Fullpipeline.py to create a jupyter notebook with cells

# In[1]
print("Hello from SageMaker")

# In[2]
%%capture
!pip install -U sagemaker s3fs h5py wandb[launch]

# In[3]
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline_context import PipelineSession

# In[4]
# WandB login
!wandb login API_KEY

# In[5]
# Configuration for SageMaker Pipeline
prefix = "pipeline-model-example"
pipeline_name = "serial-inference-pipeline"
raw_s3 = "s3://das-samples-uploader/DS_Ramsbrook_DAS_data/2024-04-18/100_lpm_60s_1"
preprocessing_machine = "ml.m5.12xlarge"
training_machine = "ml.m5.12xlarge"
validation_machine = "ml.m5.12xlarge"

# Hyperparameters
training_epochs = "10"
accuracy_threshold = 0.75
tensorflow_version = "2.4.1"
python_version = "py37"
DAS_data_channels = 376

# In[6]
# Set up SageMaker and pipeline session
sess = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=sess)
role = get_execution_role()
pipeline_session = PipelineSession()
bucket = sagemaker_session.default_bucket()
region = boto3.Session().region_name

print(region, bucket)

# In[7]
# Parameters for SageMaker Workflow
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat

input_data = ParameterString(name="InputData", default_value=raw_s3)
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=preprocessing_machine)
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
training_instance_type = ParameterString(name="TrainingInstanceType", default_value=training_machine)
training_epochs_param = ParameterString(name="TrainingEpochs", default_value=training_epochs)
accuracy_mse_threshold = ParameterFloat(name="AccuracyMseThreshold", default_value=accuracy_threshold)

# In[8]
!mkdir -p code

# In[9]
%%writefile code/preprocess.py
# Preprocess Data Script
import os
from my_package.read_DAS_hdf5 import (
    load_multi_DAS_file, list_hdf5_files_in_dir, generate_training_set_spectrogram
)
import numpy as np
from sklearn.model_selection import train_test_split

base_dir = "/opt/ml/processing"

if __name__ == "__main__":
    data_location = f"{base_dir}/input/"
    file_names = list_hdf5_files_in_dir(data_location)
    DAS_filtered_data = load_multi_DAS_file(file_names, channels=376)
    
    # Generate datasets
    training_data_leak = generate_training_set_spectrogram(
        DAS_filtered_data, 188, 199, [300000, 2182000], 20000
    )
    training_data_noleak = generate_training_set_spectrogram(
        DAS_filtered_data, 148, 270, [0, 200000], 20000
    )

    # Combine and save datasets
    training_data = np.concatenate([training_data_leak, training_data_noleak])
    training_label = np.concatenate([np.ones(len(training_data_leak)), np.zeros(len(training_data_noleak))])
    
    X_train, X_test, y_train, y_test = train_test_split(
        training_data, training_label, test_size=0.33, random_state=7
    )
    np.save(f"{base_dir}/train/X_train.npy", X_train)
    np.save(f"{base_dir}/train/y_train.npy", y_train)
    np.save(f"{base_dir}/test/X_test.npy", X_test)
    np.save(f"{base_dir}/test/y_test.npy", y_test)

# In[10]
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Processing Step Definition
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=preprocessing_machine,
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session,
)

processor_args = sklearn_processor.run(
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ProcessingInput(source="code/", destination="/opt/ml/processing/depend/code"),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
    code="code/preprocess.py",
)

# In[11]
from sagemaker.workflow.steps import ProcessingStep
step_process = ProcessingStep(name="PreprocessData", step_args=processor_args)

print(bucket, prefix)

# In[12]
%%writefile code/train.py
# Model Training Script
import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Input, Sequential
from tensorflow.keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    return parser.parse_args()

def get_data(data_dir, type="train"):
    x_data = np.load(f"{data_dir}/X_{type}.npy")
    y_data = np.load(f"{data_dir}/y_{type}.npy")
    return x_data, y_data

def get_model(num_classes):
    model = Sequential([
        Input(shape=(5, 29, 89, 1)),
        layers.Conv3D(128, (3, 3, 3), activation='relu', bias_initializer=Constant(0.01)),
        layers.Conv3D(128, (3, 3, 3), activation='relu', bias_initializer=Constant(0.01)),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    args = parse_args()
    x_train, y_train = get_data(args.train)
    x_test, y_test = get_data(args.test)

    model = get_model(num_classes=2)
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_test, y_test))
    model.save(f"{args.sm_model_dir}/model")

# In[13]
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

hyperparameters = {"epochs": training_epochs, "learning_rate": 0.1, "batch_size": 64}
model_path = f"s3://{bucket}/{prefix}/model/"

tf2_estimator = TensorFlow(
    source_dir="code",
    entry_point="train.py",
    instance_type=training_instance_type,
    instance_count=1,
    framework_version=tensorflow_version,
    role=role,
    hyperparameters=hyperparameters,
    output_path=model_path,
    py_version=python_version,
    sagemaker_session=pipeline_session,
)

train_args = tf2_estimator.fit(
    inputs={
        "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri),
        "test": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri),
    }
)

step_train_model = TrainingStep(name="TrainTensorflowModel", step_args=train_args)

# In[14]
from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[input_data, model_approval_status, training_epochs_param, accuracy_mse_threshold],
    steps=[step_process, step_train_model],
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

execution.describe()
execution.wait()
execution.list_steps()