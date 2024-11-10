import subprocess
import sys
import os
import json
import sys
import numpy as np
import pandas as pd
import pathlib
import tarfile
from sklearn.preprocessing import LabelEncoder

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/depend/code/requirements.txt",
])

from keras.utils import np_utils


if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
    import tensorflow as tf

    model = tf.keras.models.load_model("./model/1.keras")
    
    test_path = "/opt/ml/processing/test/"
    
    x_test = np.load(test_path+"/X_test.npy")
    y_test = np.load(test_path+"/y_test.npy")
    
    region = "eu-central-1"
    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_Y = encoder.transform(y_test)
    hot_y_test = np_utils.to_categorical(encoded_Y)
    
    scores = model.evaluate(x_test, hot_y_test, verbose=2)
    print("Loss, Accuracy :", scores)

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "classification_metrics": {
            "Accuracy": {"value": scores, "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
