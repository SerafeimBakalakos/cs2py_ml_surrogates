import json
import sys
import traceback

import keras
import numpy as np
from numpy._typing import NDArray

import tensorflow as tf

from src.my_utilities import arrayIO
from src.my_utilities import read_keras_network

# def train_model(features: NDArray, labels: NDArray):
#     """
#     :param features: An (m, n) array, where m = the number of samples and n = number of features per sample
#     :param labels: An (m, n) array, where m = the number of samples and n = number of labels per sample
#     :param seed: If seed != -1, TensorFlow will be configured to use the provided seed value for all random number
#         generations, in order to obtain reproducible results.
#     :return: The trained keras Sequential model
#     """
#     num_features = features.shape[1]
#     num_labels = labels.shape[1]
#
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(units=num_labels, input_shape=(num_features,))
#     ])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
#         loss='mean_absolute_error'
#     )
#     # model.summary()
#     # print('Input size=', model.input.shape)
#     history = model.fit(features, labels, epochs=100, verbose=0)
#     return model


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

if __name__ == '__main__':
    try:
        # path_settings = sys.argv[1]
        # path_results = sys.argv[2]

        path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\2024-7-21-2311_f2f55081-4883-4256-8918-fef7e44fe583_cs2py_settings.json"
        path_results = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\2024-7-21-2311_f70cdd0b-90ab-489b-9d05-5b425ac88bfd_py2cs_results.json"

        # Read settings file
        with open(path_settings) as file_set:
            settings = json.load(file_set)

        # Read and apply TensorFlow settings
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

        # Load input arrays
        train_model_params = arrayIO.load_array2D(path_train_model_params, np_dtype)
        train_solutions = arrayIO.load_array2D(path_train_solution_vectors, np_dtype)

        # Read CAE and FFNN models
        model_architecture = settings["ModelArchitecture"]
        encoder_model = read_keras_network.create_model_sequential(model_architecture["EncoderLayers"])
        decoder_model = read_keras_network.create_model_sequential(model_architecture["DecoderLayers"])
        ffnn_model = read_keras_network.create_model_sequential(model_architecture["FfnnLayers"])

        # Train and save CAE and FFNN models
        cae_history = train_cae(model_architecture, encoder_model, decoder_model, train_solutions)
        decoder_model.save(path_model_decoder)
        ffnn_history = train_ffnn(model_architecture, ffnn_model, encoder_model, train_model_params, train_solutions)
        ffnn_model.save(path_model_ffnn)

    except:  # this catches everything, unlike 'except Exception as ex'
        with open(path_results, 'w') as f:
            traceback.print_exc(file=f)
        sys.exit(100)
    else:
        sys.exit(0)