import json
import sys
import traceback

import numpy as np

import tensorflow as tf

from src.my_utilities import arrayIO


if __name__ == '__main__':
    try:
        path_settings = sys.argv[1]
        path_results = sys.argv[2]

        # Read settings file
        with open(path_settings) as file_set:
            settings = json.load(file_set)
        path_features = settings["FeaturesPath"]
        path_labels = settings["LabelsPath"]
        path_model = settings["ModelPath"]
        use_float64 = settings['Float64']
        np_dtype = np.double if use_float64 is True else np.single
        # path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\settings.json"
        # path_input = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\x.txt"
        # path_output = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\y.txt"
        # path_model = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\model.keras"

        # Load input array
        x = arrayIO.load_row_matrix(path_features, np_dtype)

        # Use model to predict
        if use_float64:
            tf.keras.backend.set_floatx('float64')
        model = tf.keras.models.load_model(path_model)
        y = model.predict(x)

        # Save output array
        arrayIO.save_row_matrix(y, path_labels)
    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_results, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(100)
    else:
        sys.exit(0)