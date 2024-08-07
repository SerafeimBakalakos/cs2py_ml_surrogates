import time

import numpy as np
import tensorflow as tf

from src.my_utilities import arrayIO, csharp_interop


def run_main_script_body(settings: dict, durations: dict):
    # Read and apply TensorFlow settings
    start = time.time_ns()
    path_model_params = settings["ModelParamsPath"]
    path_solution_vector = settings["SolutionVectorPath"]
    path_model_decoder = settings["ModelDecoderPath"]
    path_model_ffnn = settings["ModelFfnnPath"]

    use_float64 = settings["Float64"]
    if use_float64:
        tf.keras.backend.set_floatx('float64')
    np_dtype = np.double if use_float64 is True else np.single

    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["Setup"] = durations["Setup"] + elapsed


    # Load model and input array
    start = time.time_ns()
    decoder_model = tf.keras.models.load_model(path_model_decoder)
    ffnn_model = tf.keras.models.load_model(path_model_ffnn)
    x = arrayIO.load_row_matrix(path_model_params, np_dtype)
    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["IO"] = durations["IO"] + elapsed

    # Use model to predict
    start = time.time_ns()
    temp = ffnn_model.predict(x)
    y = decoder_model.predict(temp)
    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["Actual"] = durations["Actual"] + elapsed

    # Save output array
    start = time.time_ns()
    arrayIO.squeeze_and_save_tensor(y, path_solution_vector)
    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["IO"] = durations["IO"] + elapsed


if __name__ == '__main__':
    # For testing:
    import sys
    # work_directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\testing_python"
    # work_directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear"
    # path_settings = work_directory + "\\predict_cs2py_settings.json"
    # path_results = work_directory + "\\predict_py2cs_results.json"
    # path_log = work_directory + "\\predict_py2cs_log.json"
    # path_settings = work_directory + "\\2024-8-6-122_d328438b-6f5c-47c4-ad0b-58b296cd8011_cs2py_settings.json"
    # path_results = work_directory + "\\2024-8-6-122_d328438b-6f5c-47c4-ad0b-58b296cd8011_py2cs_results.json"
    # path_log = work_directory + "\\2024-8-6-122_d328438b-6f5c-47c4-ad0b-58b296cd8011_py2cs_log.json"
    # sys.argv = [sys.argv[0]] + [path_settings, path_results, path_log]

    # Actual script
    csharp_interop.call_csharp_script(run_main_script_body)

