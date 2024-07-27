import numpy as np

import tensorflow as tf

from timeit import default_timer as timer
from src.my_utilities import arrayIO, read_keras_network, csharp_interop


def train_cae(model_architecture, encoder_model, decoder_model, solution_vectors):
    # Read the relevant properties
    num_dofs = model_architecture["NumDofs"]
    cae_batch_size = model_architecture["CaeBatchSize"]
    cae_num_epochs = model_architecture["CaeNumEpochs"]
    cae_learning_rate = model_architecture["CaeLearningRate"]

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
    ffnn_learning_rate = model_architecture["FfnnLearningRate"]

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


def run_main_script_body(settings: dict, durations: dict):
    # Read and apply TensorFlow settings
    start = timer()
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

    elapsed = timer() - start
    durations["setup"] = durations["setup"] + elapsed

    # Load input arrays
    start = timer()
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

    elapsed = timer() - start
    durations["IO"] = durations["IO"] + elapsed

    # Train CAE and FFNN models
    start = timer()
    print("***** Training CAE *****")
    cae_history = train_cae(model_architecture, encoder_model, decoder_model, train_solutions)
    print("***** Training FFNN *****")
    ffnn_history = train_ffnn(model_architecture, ffnn_model, encoder_model, train_model_params, train_solutions)

    elapsed = timer() - start
    durations["actual"] = durations["actual"] + elapsed

    # Save CAE and FFNN models
    start = timer()
    decoder_model.save(path_model_decoder)
    ffnn_model.save(path_model_ffnn)
    elapsed = timer() - start
    durations["IO"] = durations["IO"] + elapsed


if __name__ == '__main__':
    # For testing:
    # import sys
    # path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\testing_python\\train_cs2py_settings.json"
    # path_results = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\testing_python\\train_py2cs_results.json"
    # sys.argv = [sys.argv[0]] + [path_settings, path_results]

    # Actual script
    csharp_interop.call_csharp_script(run_main_script_body)