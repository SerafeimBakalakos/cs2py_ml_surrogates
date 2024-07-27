import numpy as np

import tensorflow as tf

from timeit import default_timer as timer
from src.my_utilities import arrayIO, csharp_interop


def run_main_script_body(settings: dict, durations: dict):
    # Read and apply TensorFlow settings
    start = timer()
    path_model_params = settings["ModelParamsPath"]
    path_solution_vector = settings["SolutionVectorPath"]
    path_model_decoder = settings["ModelDecoderPath"]
    path_model_ffnn = settings["ModelFfnnPath"]

    use_float64 = settings["Float64"]
    if use_float64:
        tf.keras.backend.set_floatx('float64')
    np_dtype = np.double if use_float64 is True else np.single

    elapsed = timer() - start
    durations["setup"] = durations["setup"] + elapsed


    # Load model and input array
    start = timer()
    decoder_model = tf.keras.models.load_model(path_model_decoder)
    ffnn_model = tf.keras.models.load_model(path_model_ffnn)
    x = arrayIO.load_row_matrix(path_model_params, np_dtype)
    elapsed = timer() - start
    durations["IO"] = durations["IO"] + elapsed

    # Use model to predict
    start = timer()
    temp = ffnn_model.predict(x)
    y = decoder_model.predict(temp)
    elapsed = timer() - start
    durations["actual"] = durations["actual"] + elapsed

    # Save output array
    start = timer()
    arrayIO.squeeze_and_save_tensor(y, path_solution_vector)
    elapsed = timer() - start
    durations["IO"] = durations["IO"] + elapsed


if __name__ == '__main__':
    # For testing:
    # import sys
    # path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\testing_python\\predict_cs2py_settings.json"
    # path_results = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\testing_python\\predict_py2cs_results.json"
    # sys.argv = [sys.argv[0]] + [path_settings, path_results]

    # Actual script
    csharp_interop.call_csharp_script(run_main_script_body)