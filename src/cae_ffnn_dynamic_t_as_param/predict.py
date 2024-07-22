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

        path_model_params = settings["ModelParamsPath"]
        path_solution_vector = settings["SolutionVectorPath"]
        path_model_decoder = settings["ModelDecoderPath"]
        path_model_ffnn = settings["ModelFfnnPath"]

        use_float64 = settings["Float64"]
        if use_float64:
            tf.keras.backend.set_floatx('float64')
        np_dtype = np.double if use_float64 is True else np.single

        # Load input array
        x = arrayIO.load_row_matrix(path_model_params, np_dtype)

        # Use model to predict
        decoder_model = tf.keras.models.load_model(path_model_decoder)
        ffnn_model = tf.keras.models.load_model(path_model_ffnn)
        temp = ffnn_model.predict(x)
        y = decoder_model.predict(temp)

        # Save output array
        arrayIO.save_row_matrix(y, path_solution_vector)
    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_results, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(100)
    else:
        sys.exit(0)