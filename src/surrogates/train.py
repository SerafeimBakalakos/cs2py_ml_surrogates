import json
import sys
import traceback

import numpy as np
from numpy._typing import NDArray

import tensorflow as tf

from src.my_utilities import arrayIO

def train_model(features: NDArray, labels: NDArray, seed: int, use_float64: bool):
    """
    :param features: An (m, n) array, where m = the number of samples and n = number of features per sample
    :param labels: An (m, n) array, where m = the number of samples and n = number of labels per sample
    :param use_float64: If true, TensorFlow will be configured for double precision (64bit floats) numbers.
    :param seed: If seed != -1, TensorFlow will be configured to use the provided seed value for all random number
        generations, in order to obtain reproducible results.
    :return: The trained keras Sequential model
    """
    num_features = features.shape[1]
    num_labels = labels.shape[1]

    if seed != -1:
        tf.keras.utils.set_random_seed(seed)
    if use_float64:
        tf.keras.backend.set_floatx('float64')
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=num_labels, input_shape=(num_features,))
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )
    # model.summary()
    # print('Input size=', model.input.shape)
    history = model.fit(features, labels, epochs=100, verbose=0)
    return model


if __name__ == '__main__':
    try:
        path_settings = sys.argv[1]
        path_results = sys.argv[2]

        # Read settings file
        with open(path_settings) as file_set:
            settings = json.load(file_set)
        path_train_features = settings["FeaturesPath"]
        path_train_labels = settings["LabelsPath"]
        path_model = settings["ModelPath"]
        tf_use_float64 = settings["Float64"]
        np_dtype = np.double if tf_use_float64 is True else np.single
        tf_seed = settings["TensorFlowSeed"]
        # path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\settings.json"
        # path_train_features = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\features.txt"
        # path_train_labels = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\labels.txt"
        # path_model = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\model.keras"

        # Load input arrays
        x = arrayIO.load_array2D(path_train_features, np_dtype)
        y = arrayIO.load_array2D(path_train_labels, np_dtype)

        # Train model
        tf_model = train_model(x, y, tf_seed, tf_use_float64)
        tf_model.save(path_model)
    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_results, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(100)
    else:
        sys.exit(0)