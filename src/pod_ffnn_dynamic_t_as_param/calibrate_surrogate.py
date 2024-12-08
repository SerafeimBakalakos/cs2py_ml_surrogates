import keras.optimizers.schedules
import numpy as np
import tensorflow as tf

from src.my_utilities import arrayIO


# Returns the tuple (ffnn_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_ffnn(num_train_samples:int, num_model_params:int, num_pod_coeffs:int, train_model_params, train_pod_coeffs,
                load_ffnn:bool, save_ffnn:bool, directory:str):
    path_ffnn = directory + "\\model_ffnn_43.keras"

    if (load_ffnn):
        print("\nReading FFNN model from disc")
        ffnn_model = tf.keras.models.load_model(path_ffnn)
        lst_loss_history = []
        return (ffnn_model, lst_loss_history)
    else:
        (ffnn_model, lst_loss_history) = train_ffnn(
            num_train_samples, num_model_params, num_pod_coeffs, train_model_params, train_pod_coeffs)
        if (save_ffnn):
            ffnn_model.save(path_ffnn, lst_loss_history)
        return (ffnn_model, lst_loss_history)


def train_ffnn(num_train_samples:int, num_model_params:int, num_pod_coeffs:int, train_model_params, train_pod_coeffs):
    # Training properties
    ffnn_batch_size = 20
    ffnn_num_epochs = 5000
    ffnn_hidden_size = 64
    ffnn_shuffle = True
    ffnn_activation = 'relu'
    #ffnn_activation = 'tanh'

    # Learning rate
    ffnn_learning_rate = provide_learning_rate_schedule(
        initial_learning_rate=1E-3, final_learning_rate=1E-5, staircase=True,
        num_epochs=ffnn_num_epochs, batch_size=ffnn_batch_size, num_training_samples=num_train_samples)

    # Architecture
    ffnn_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=num_model_params),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(ffnn_hidden_size),
        provide_activation_func(ffnn_activation),
        tf.keras.layers.Dense(num_pod_coeffs)
    ])

    # Compile
    ffnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ffnn_learning_rate), loss='mse')
    ffnn_output = train_pod_coeffs

    # Fit
    print("\nTraining FFNN")
    history = ffnn_model.fit(train_model_params, ffnn_output,
                             batch_size=ffnn_batch_size, epochs=ffnn_num_epochs, shuffle=ffnn_shuffle)
    print("_________________________________________________________________")

    return (ffnn_model, history.history['loss'])


def provide_activation_func(func_name:str):
    if func_name == 'relu':
        #print("using relu")
        return tf.keras.layers.LeakyReLU()
    elif func_name == 'tanh':
        return tf.keras.layers.Activation(tf.keras.activations.tanh)
    else:
        raise Exception('Invalid activation')

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

def read_datasets(directory:str):
    path_train_parameters = directory + "\\train_model_params.npy"
    path_train_pod_coeffs = directory + "\\train_pod_coeffs.npy"
    path_test_parameters = directory + "\\test_model_params.npy"
    path_test_pod_coeffs = directory + "\\test_pod_coeffs.npy"

    train_model_params = arrayIO.load_array2D(path_train_parameters, np.single)
    train_pod_coeffs = arrayIO.load_array2D(path_train_pod_coeffs, np.single)

    test_model_params = arrayIO.load_array2D(path_test_parameters, np.single)
    test_pod_coeffs = arrayIO.load_array2D(path_test_pod_coeffs, np.single)

    assert train_model_params.shape[0] == train_pod_coeffs.shape[0]
    assert test_model_params.shape[0] == test_pod_coeffs.shape[0]
    assert train_model_params.shape[1] == test_model_params.shape[1]
    assert train_pod_coeffs.shape[1] == test_pod_coeffs.shape[1]

    return (train_model_params, train_pod_coeffs, test_model_params, test_pod_coeffs)


def calc_vector_error_entrywise_max_absolute(expected, predicted):
    num_entries = expected.shape[0]
    max_error = -1;
    current_error = -1;
    for i in range(num_entries):
        if expected[i] != 0:
            current_error = abs((expected[i] - predicted[i]) / expected[i])
        else:
            current_error = abs(expected[i] - predicted[i])
        max_error = max(max_error, current_error)
    return max_error

def calc_vector_error_entrywise_mean_absolute(expected, predicted):
    num_entries = expected.shape[0]
    total_error = 0;
    for i in range(num_entries):
        if expected[i] != 0:
            total_error += abs((expected[i] - predicted[i]) / expected[i])
        else:
            total_error += abs(expected[i] - predicted[i])
    return total_error / num_entries


def calc_vector_error_normwise(expected, predicted):
    return np.linalg.norm(expected - predicted) / np.linalg.norm(expected)


def test_ffnn(ffnn_model, test_model_params, test_pod_coeffs):
    num_test_samples = test_model_params.shape[0]
    coeff_predictions = ffnn_model(test_model_params)
    #coeff_predictions = np.squeeze(coeff_predictions)
    #test_pod_coeffs = np.squeeze(test_pod_coeffs)
    mean_error = 0
    for s in range(num_test_samples):
        expected = test_pod_coeffs[s:s+1, :]
        predicted = coeff_predictions[s:s+1, :]
        mean_error += calc_vector_error_normwise(expected, predicted)
        #mean_error += calc_vector_error_entrywise_max_absolute(expected, predicted)
        #mean_error += calc_vector_error_entrywise_mean_absolute(expected, predicted)
    mean_error /= num_test_samples
    return mean_error



if __name__ == '__main__':
    # Run constants
    save_ffnn = True
    load_ffnn = False
    tf.keras.utils.set_random_seed(23)


    # Read datasets from disc
    directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\python_experimenting"
    #directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear\\python_experimenting"
    (train_model_params, train_pod_coeffs, test_model_params, test_pod_coeffs) = read_datasets(directory)
    num_pod_coeffs = train_pod_coeffs.shape[1]
    num_model_params = train_model_params.shape[1]
    num_train_samples = train_pod_coeffs.shape[0]
    if (train_model_params.shape[0] != num_train_samples):
        raise Exception("The number of training samples must be the same in the model parameters and the pod coefficients datasets")

    # Train (or read from disc) the networks
    (ffnn_model, loss_ffnn) = create_ffnn(num_train_samples, num_model_params, num_pod_coeffs, train_model_params,
                                          train_pod_coeffs, load_ffnn, save_ffnn, directory)

    # Test surrogate
    print('_________________________________________________________________')
    print('Testing model')
    mean_error_ffnn = 0
    mean_error_ffnn = test_ffnn(ffnn_model, test_model_params, test_pod_coeffs)

    # Print results
    print('_________________________________________________________________')
    if len(loss_ffnn) > 0:
        print("FFNN loss function: at start = " + str(loss_ffnn[0]) + " - at end = " + str(loss_ffnn[-1]))
    if mean_error_ffnn > 0:
        print('FFNN mean error on test set (|expected - predicted| / |expected| = ' + str(mean_error_ffnn))

