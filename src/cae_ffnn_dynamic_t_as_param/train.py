import time

import numpy as np
import tensorflow as tf

from src.my_utilities import arrayIO, read_keras_network, csharp_interop


def train_cae(model_architecture, encoder_model, decoder_model, solution_vectors):
    # Read the relevant properties
    num_dofs = model_architecture["NumDofs"]
    cae_batch_size = model_architecture["CaeBatchSize"]
    cae_num_epochs = model_architecture["CaeNumEpochs"]
    cae_learning_rate_start = model_architecture["CaeLearningRateStart"]
    cae_learning_rate_end = model_architecture["CaeLearningRateEnd"]

    # Learning rate schedule
    cae_learning_rate = provide_learning_rate_schedule(
        initial_learning_rate=cae_learning_rate_start, final_learning_rate=cae_learning_rate_end, staircase=True,
        num_epochs=cae_num_epochs, batch_size=cae_batch_size, num_training_samples=solution_vectors.shape[0])

    # Compile
    cae_input = tf.keras.Input(shape=(1, num_dofs))
    encoded_input = encoder_model(cae_input)
    decoded_output = decoder_model(encoded_input)
    cae_model = tf.keras.Model(cae_input, decoded_output)
    cae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cae_learning_rate), loss='mse')

    # Fit
    history = cae_model.fit(solution_vectors, solution_vectors, batch_size=cae_batch_size, epochs=cae_num_epochs,
                            shuffle=True)
    return history


def train_ffnn(model_architecture, ffnn_model, encoder_model, model_params, solution_vectors):
    # Read the relevant properties
    ffnn_batch_size = model_architecture["FfnnBatchSize"]
    ffnn_num_epochs = model_architecture["FfnnNumEpochs"]
    ffnn_learning_rate_start = model_architecture["FfnnLearningRateStart"]
    ffnn_learning_rate_end = model_architecture["FfnnLearningRateEnd"]

    # Learning rate schedule
    ffnn_learning_rate = provide_learning_rate_schedule(
        initial_learning_rate=ffnn_learning_rate_start, final_learning_rate=ffnn_learning_rate_end, staircase=True,
        num_epochs=ffnn_num_epochs, batch_size=ffnn_batch_size, num_training_samples=solution_vectors.shape[0])

    # Compile
    ffnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ffnn_learning_rate), loss='mse')
    ffnn_output = encoder_model(solution_vectors)

    # Fit
    history = ffnn_model.fit(model_params, ffnn_output, batch_size=ffnn_batch_size, epochs=ffnn_num_epochs,
                             shuffle=True)
    return history


def print_to_file(filepath: str, msg: str):
    with open(filepath, "a'") as file:
        file.write(msg)

def provide_learning_rate_schedule(initial_learning_rate:float, final_learning_rate:float, staircase:bool,
                                    num_epochs:int, batch_size:int, num_training_samples:int):
    if initial_learning_rate < final_learning_rate:
        raise Exception('Initial learning rate must not be less than the final one')
    elif initial_learning_rate == final_learning_rate:
        return initial_learning_rate
    else:
        decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
        steps_per_epoch = int(num_training_samples / batch_size)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate, decay_steps=steps_per_epoch,
            decay_rate=decay_rate, staircase=staircase)
        return lr_schedule

def run_main_script_body(settings: dict, durations: dict):
    # Read and apply TensorFlow settings
    start = time.time_ns()
    print("Reading surrogate parameters")
    tf_seed = settings["TensorFlowSeed"]
    if tf_seed != -1:
        tf.keras.utils.set_random_seed(tf_seed)
    tf_use_float64 = settings["Float64"]
    np_dtype = np.double if tf_use_float64 is True else np.single
    if tf_use_float64:
        tf.keras.backend.set_floatx('float64')

    # Read the paths for files used to communicate data
    path_train_model_params = settings["TrainModelParamsPath"]
    path_train_solution_vectors = settings["TrainSolutionVectorsPath"]
    # path_model_encoder = settings["ModelEncoderPath"]
    path_model_decoder = settings["ModelDecoderPath"]
    path_model_ffnn = settings["ModelFfnnPath"]

    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["Setup"] = durations["Setup"] + elapsed

    # Load input arrays
    start = time.time_ns()
    print("Loading training data")
    train_model_params = arrayIO.load_array2D(path_train_model_params, np_dtype)
    train_solutions = arrayIO.load_array2D(path_train_solution_vectors, np_dtype)
    train_solutions = np.expand_dims(train_solutions, axis=1)

    # Read CAE and FFNN models
    print("Setting up keras models")
    model_architecture = settings["ModelArchitecture"]
    encoder_model = read_keras_network.create_model_sequential(model_architecture["EncoderLayers"])
    decoder_model = read_keras_network.create_model_sequential(model_architecture["DecoderLayers"])
    ffnn_model = read_keras_network.create_model_sequential(model_architecture["FfnnLayers"])

    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["IO"] = durations["IO"] + elapsed

    # Train CAE and FFNN models
    start = time.time_ns()
    print("***** Training CAE *****")
    cae_history = train_cae(model_architecture, encoder_model, decoder_model, train_solutions)
    print("***** Training FFNN *****")
    ffnn_history = train_ffnn(model_architecture, ffnn_model, encoder_model, train_model_params, train_solutions)

    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["Actual"] = durations["Actual"] + elapsed

    # Save CAE and FFNN models
    start = time.time_ns()
    decoder_model.save(path_model_decoder)
    ffnn_model.save(path_model_ffnn)
    elapsed = (time.time_ns() - start) // 1000000 # in ms
    durations["IO"] = durations["IO"] + elapsed


if __name__ == '__main__':
    # For testing:
    # import sys
    # work_directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear"
    # work_directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear"
    # path_settings = work_directory + "\\train_cs2py_settings.json"
    # path_results = work_directory + "\\train_py2cs_results.json"
    # path_log = work_directory + "\\train_py2cs_log.json"
    # path_settings = work_directory + "\\2024-8-9-15_80c66b64-e061-4154-a81a-23e10e50d7b3_cs2py_settings.json"
    # path_results = work_directory + "\\2024-8-9-15_80c66b64-e061-4154-a81a-23e10e50d7b3_py2cs_results.json"
    # path_log = work_directory + "\\2024-8-9-15_80c66b64-e061-4154-a81a-23e10e50d7b3_py2cs_log.json"
    # sys.argv = [sys.argv[0]] + [path_settings, path_results, path_log]

    # Actual script
    csharp_interop.call_csharp_script(run_main_script_body)