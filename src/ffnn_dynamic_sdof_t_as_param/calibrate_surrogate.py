import keras.optimizers.schedules
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.my_utilities import arrayIO

loss_is_mse = True

# Returns the tuple (ffnn_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_ffnn(num_train_samples:int, num_model_params:int, num_dofs:int, train_model_params, train_solutions,
                load_ffnn:bool, save_ffnn:bool, directory:str, validation_model_params, validation_solutions):
    path_ffnn = directory + "\\model_ffnn_43.keras"

    if (load_ffnn):
        print("\nReading FFNN model from disc")
        ffnn_model = tf.keras.models.load_model(path_ffnn)
        history = None
        return (ffnn_model, history)
    else:
        (ffnn_model, history) = train_ffnn(
            num_train_samples, num_model_params, num_dofs, train_model_params, train_solutions,
            validation_model_params, validation_solutions)
        if (save_ffnn):
            ffnn_model.save(path_ffnn, history.history['loss'])
        return (ffnn_model, history)

def train_ffnn(num_train_samples:int, num_model_params:int, num_dofs:int, train_model_params, train_solutions,
               validation_model_params, validation_solutions):
    # Training properties
    ffnn_batch_size = 100
    ffnn_num_epochs = 5000
    ffnn_hidden_size = 64
    ffnn_shuffle = True
    #ffnn_activation = 'relu'
    ffnn_activation = 'tanh'
    ffnn_loss_fun = 'mse' # options 'mse' (default), 'mape'
    ffnn_lr_decay_epochs = 5000

    # Learning rate
    ffnn_learning_rate = provide_learning_rate_schedule(
        initial_learning_rate=1E-6, final_learning_rate=1E-8, staircase=True,
        num_epochs=ffnn_lr_decay_epochs, batch_size=ffnn_batch_size, num_training_samples=num_train_samples)

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
        #tf.keras.layers.Dense(ffnn_hidden_size),
        #provide_activation_func(ffnn_activation),
        #tf.keras.layers.Dense(ffnn_hidden_size),
        #provide_activation_func(ffnn_activation),

        tf.keras.layers.Dense(num_dofs)
    ])

    # Compile
    ffnn_metrics = []
    global loss_is_mse
    if ffnn_loss_fun == 'mse': #is tf.keras.losses.MeanSquaredError:
        ffnn_metrics = [keras.src.metrics.MeanAbsolutePercentageError()]
        loss_is_mse = True
    elif ffnn_loss_fun == 'mape': #is tf.keras.losses.MeanAbsolutePercentageError:
        ffnn_metrics = [keras.src.metrics.MeanSquaredError()]
        loss_is_mse = False
    else:
        ffnn_metrics = [keras.src.metrics.MeanSquaredError(), keras.src.metrics.MeanAbsolutePercentageError()]
        loss_is_mse = False

    ffnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ffnn_learning_rate), loss=ffnn_loss_fun,
                       metrics=ffnn_metrics)
    ffnn_output = train_solutions

    # Fit
    print("\nTraining FFNN")
    if (validation_model_params is not None) and (validation_solutions is not None):
        history = ffnn_model.fit(
            train_model_params, ffnn_output, batch_size=ffnn_batch_size, epochs=ffnn_num_epochs, shuffle=ffnn_shuffle,
            validation_data=(validation_model_params, validation_solutions))
    else:
        history = ffnn_model.fit(train_model_params, ffnn_output,
                                 batch_size=ffnn_batch_size, epochs=ffnn_num_epochs, shuffle=ffnn_shuffle)
    print("_________________________________________________________________")


    return (ffnn_model, history)


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

def read_all_datasets(directory:str):
    path_train_parameters = directory + "\\train_model_params.npy"
    path_train_solutions = directory + "\\train_solutions.npy"
    path_test_parameters = directory + "\\test_model_params.npy"
    path_test_solutions = directory + "\\test_solutions.npy"
    path_validation_parameters = directory + "\\validation_model_params.npy"
    path_validation_solutions = directory + "\\validation_solutions.npy"

    train_model_params = arrayIO.load_array2D(path_train_parameters, np.single)
    train_solutions = arrayIO.load_array2D(path_train_solutions, np.single)
    test_model_params = arrayIO.load_array2D(path_test_parameters, np.single)
    test_solutions = arrayIO.load_array2D(path_test_solutions, np.single)
    validation_model_params = arrayIO.load_array2D(path_validation_parameters, np.single)
    validation_solutions = arrayIO.load_array2D(path_validation_solutions, np.single)

    assert train_model_params.shape[0] == train_solutions.shape[0]
    assert test_model_params.shape[0] == test_solutions.shape[0]
    assert validation_model_params.shape[0] == validation_solutions.shape[0]
    assert train_model_params.shape[1] == test_model_params.shape[1]
    assert train_model_params.shape[1] == validation_model_params.shape[1]
    assert train_solutions.shape[1] == test_solutions.shape[1]
    assert train_solutions.shape[1] == validation_solutions.shape[1]

    return (train_model_params, train_solutions, test_model_params, test_solutions, validation_model_params, validation_solutions)


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


def test_ffnn(ffnn_model, test_model_params, test_solutions):
    num_test_samples = test_model_params.shape[0]
    coeff_predictions = ffnn_model(test_model_params)
    #coeff_predictions = np.squeeze(coeff_predictions)
    #test_pod_coeffs = np.squeeze(test_pod_coeffs)
    mean_error = 0
    num_non_zeros = 0
    for s in range(num_test_samples):
        expected = test_solutions[s:s+1, :]
        #if expected[0] == 0:
        #    print("Zero at " + str(s))
        predicted = coeff_predictions[s:s+1, :]
        err = calc_vector_error_normwise(expected, predicted)
        #err = calc_vector_error_entrywise_max_absolute(expected, predicted)
        #err = calc_vector_error_entrywise_mean_absolute(expected, predicted)
        if expected != 0:
            mean_error += err
            num_non_zeros += 1
    mean_error /= num_non_zeros
    if num_non_zeros < num_test_samples:
        print("There are " + str(num_test_samples - num_non_zeros) + " zero samples")
    return mean_error

def plot_error_history(history):
    if (loss_is_mse):
        title_1 = 'Mean Squared Error'
        series_1 = 'loss'
        title_2 = 'Mean Absolute Percentage Error'
        series_2 = 'mean_absolute_percentage_error'
    else:
        title_1 = 'Mean Absolute Percentage Error'
        series_1 = 'loss'
        title_2 = 'Mean Squared Error'
        series_2 = 'mean_squared_error'

    # Plot error history
    plt.figure()
    plt.plot(history.history[series_1])
    plt.plot(history.history['val_' + series_1], linestyle="dashed")
    plt.title('FFNN error history')
    plt.xlabel('epoch')
    plt.ylabel(title_1)
    plt.yscale('log')
    plt.legend(['train set', 'validation set'], loc='upper right')
    plt.draw()

    plt.figure()
    plt.plot(history.history[series_2])
    plt.plot(history.history['val_' + series_2],  linestyle="dashed")
    plt.title('FFNN error history')
    plt.xlabel('epoch')
    plt.ylabel(title_2)
    plt.yscale('log')
    plt.legend(['train set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    # Run constants
    save_ffnn = True
    load_ffnn = False
    tf.keras.utils.set_random_seed(23)


    # Read datasets from disc
    directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\SingleDof\\python_experimenting"
    #directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear\\python_experimenting"
    #(train_model_params, train_pod_coeffs, test_model_params, test_pod_coeffs) = read_datasets(directory)
    (train_model_params, train_solutions, test_model_params, test_solutions, validation_model_params, validation_solutions) = read_all_datasets(directory)
    num_train_samples = train_solutions.shape[0]
    num_dofs = train_solutions.shape[1]
    num_model_params = train_model_params.shape[1]
    if (train_model_params.shape[0] != num_train_samples):
        raise Exception("The number of training samples must be the same in the model parameters and the solutions datasets")

    # Train (or read from disc) the networks
    (ffnn_model, history_ffnn) = create_ffnn(
        num_train_samples, num_model_params, num_dofs, train_model_params, train_solutions, load_ffnn, save_ffnn,
        directory, validation_model_params, validation_solutions)

    # Test surrogate
    print('_________________________________________________________________')
    print('Testing model')
    mean_test_error_ffnn = test_ffnn(ffnn_model, test_model_params, test_solutions)
    mean_train_error_ffnn = test_ffnn(ffnn_model, train_model_params, train_solutions)
    mean_val_error_ffnn = test_ffnn(ffnn_model, validation_model_params, validation_solutions)

    # Print results
    print('_________________________________________________________________')
    if history_ffnn is not None:
        loss_ffnn = history_ffnn.history['loss']
        print("FFNN loss function: at start = " + str(loss_ffnn[0]) + " - at end = " + str(loss_ffnn[-1]))

    print('FFNN mean error on train set (|expected - predicted| / |expected| = ' + str(mean_train_error_ffnn))
    print('FFNN mean error on test set (|expected - predicted| / |expected| = ' + str(mean_test_error_ffnn))
    print('FFNN mean error on validation set (|expected - predicted| / |expected| = ' + str(mean_val_error_ffnn))

    plot_error_history(history_ffnn)
