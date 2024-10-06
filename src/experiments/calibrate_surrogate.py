import keras.optimizers.schedules
import numpy as np
import tensorflow as tf

from src.my_utilities import arrayIO

# Returns a tuple (encoder_model, decoder_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_cae(num_train_samples:int, num_dofs:int, latent_space_dim:int, train_solutions, load_cae:bool, save_cae:bool, directory:str):
    path_encoder = directory + "\\encoder_model.keras"
    path_decoder = directory + "\\decoder_model.keras"

    if (load_cae):
        print("\nReading CAE models from disc")
        encoder_model = tf.keras.models.load_model(path_encoder)
        decoder_model = tf.keras.models.load_model(path_decoder)
        lst_loss_history = []
        return (encoder_model, decoder_model, lst_loss_history)
    else:
        (encoder_model, decoder_model, lst_loss_history) = train_cae(
            num_train_samples, num_dofs, latent_space_dim, train_solutions)
        if (save_cae):
            encoder_model.save(path_encoder)
            decoder_model.save(path_decoder)
        return (encoder_model, decoder_model, lst_loss_history)


# Returns the tuple (ffnn_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_ffnn(num_train_samples:int, num_model_params:int, latent_space_dim:int, train_model_params, train_solutions,
                encoder_model,load_ffnn:bool, save_ffnn:bool, directory:str):
    path_ffnn = directory + "\\ffnn_model.keras"

    if (load_ffnn):
        print("\nReading FFNN model from disc")
        ffnn_model = tf.keras.models.load_model(path_ffnn)
        lst_loss_history = []
        return (ffnn_model, lst_loss_history)
    else:
        (ffnn_model, lst_loss_history) = train_ffnn(
            num_train_samples, num_model_params, latent_space_dim, train_model_params, train_solutions, encoder_model)
        if (save_ffnn):
            ffnn_model.save(path_ffnn, lst_loss_history)
        return (ffnn_model, lst_loss_history)


def train_cae(num_train_samples:int, num_dofs:int, latent_space_dim:int, train_solutions):
    # Training properties
    cae_batch_size = 20 # num_timesteps = 60
    cae_num_epochs = 500
    cae_keras_shuffle = True
    cae_kernel_size = 5
    cae_activation = 'relu'
    # cae_activation = 'tanh'

    # cae_learning_rate = 1E-4
    cae_learning_rate = provide_learning_rate_schedule(
        initial_learning_rate=1E-3, final_learning_rate=1E-4, staircase=True,
        num_epochs=cae_num_epochs, batch_size=cae_batch_size, num_training_samples=num_train_samples)

    # Network architecture
    cae_encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1, num_dofs)),
        # tf.keras.layers.Conv1D(filters=4096, kernel_size=cae_kernel_size, strides=1, padding='same'),
        # provide_activation_func(cae_activation),
        tf.keras.layers.Conv1D(filters=128, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Conv1D(filters=64, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Conv1D(filters=32, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Conv1D(filters=16, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_space_dim)
    ])

    cae_decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=latent_space_dim),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Reshape(target_shape=(1, 16)),
        tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation),
        # tf.keras.layers.Conv1DTranspose(filters=4096, kernel_size=cae_kernel_size, strides=1, padding='same'),
        # provide_activation_func(cae_activation),
        tf.keras.layers.Conv1DTranspose(filters=num_dofs, kernel_size=cae_kernel_size, strides=1, padding='same'),
        provide_activation_func(cae_activation)
    ])

    # Compile
    cae_input = tf.keras.Input(shape=(1, num_dofs))
    encoded_input = cae_encoder(cae_input)
    decoded_output = cae_decoder(encoded_input)
    cae_model = tf.keras.Model(cae_input, decoded_output)
    cae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cae_learning_rate), loss='mse')
    cae_model.summary()

    # Fit
    print("\nTraining CAE")
    history = cae_model.fit(train_solutions, train_solutions, batch_size=cae_batch_size, epochs=cae_num_epochs, shuffle=cae_keras_shuffle)
    # cae_model.summary()
    print("_________________________________________________________________")
    return (cae_encoder, cae_decoder, history.history['loss'])


def train_ffnn(num_train_samples:int, num_model_params:int, latent_space_dim:int, train_model_params, train_solutions, encoder_model):
    # Training properties
    ffnn_batch_size = 20
    ffnn_num_epochs = 5000
    ffnn_hidden_size = 64
    ffnn_shuffle = False
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
        tf.keras.layers.Dense(latent_space_dim)
    ])

    # Compile
    ffnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ffnn_learning_rate), loss='mse')
    ffnn_output = encoder_model(train_solutions)

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
    path_train_solutions = directory + "\\train_solutions.npy"
    path_test_parameters = directory + "\\test_model_params.npy"
    path_test_solutions = directory + "\\test_solutions.npy"

    train_model_params = arrayIO.load_array2D(path_train_parameters, np.single)
    train_solutions = arrayIO.load_array2D(path_train_solutions, np.single)
    train_solutions = np.expand_dims(train_solutions, axis=1)

    test_model_params = arrayIO.load_array2D(path_test_parameters, np.single)
    test_solutions = arrayIO.load_array2D(path_test_solutions, np.single)
    test_solutions = np.expand_dims(test_solutions, axis=1)

    assert train_model_params.shape[0] == train_solutions.shape[0]
    assert test_model_params.shape[0] == test_solutions.shape[0]
    assert train_model_params.shape[1] == test_model_params.shape[1]
    assert train_solutions.shape[2] == test_solutions.shape[2]

    return (train_model_params, train_solutions, test_model_params, test_solutions)


def test_cae(encoder_model, decoder_model, test_solutions):
    num_test_samples = test_solutions.shape[0]
    solution_predictions = decoder_model(encoder_model(test_solutions))
    solution_predictions = np.squeeze(solution_predictions)
    test_solutions = np.squeeze(test_solutions)
    mean_error = 0
    min_error = 1E10
    max_error = -1E10
    for s in range(num_test_samples):
        expected = np.squeeze(test_solutions[s:s + 1, :])
        predicted = np.squeeze(solution_predictions[s:s + 1, :])
        error = calc_vector_error_normwise(expected, predicted)
        #error = calc_vector_error_entrywise_mean_absolute(expected, predicted)
        #error = calc_vector_error_entrywise_max_absolute(expected, predicted)
        mean_error += error
        min_error = min(min_error, error)
        max_error = max(max_error, error)
    mean_error /= num_test_samples
    return (mean_error, min_error, max_error)

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


def test_full_surrogate(decoder_model, ffnn_model, test_model_params, test_solutions):
    num_test_samples = test_model_params.shape[0]
    solution_predictions = decoder_model(ffnn_model(test_model_params))
    solution_predictions = np.squeeze(solution_predictions)
    test_solutions = np.squeeze(test_solutions)
    mean_error = 0
    for s in range(num_test_samples):
        expected = test_solutions[s:s+1, :]
        predicted = solution_predictions[s:s+1, :]
        mean_error += np.linalg.norm(expected - predicted) / np.linalg.norm(expected)
    mean_error /= num_test_samples
    return mean_error




if __name__ == '__main__':
    # Run constants
    save_cae = True
    load_cae = False
    save_ffnn = True
    load_ffnn = False

    latent_space_dim = 8
    tf.keras.utils.set_random_seed(23)

    # Read datasets from disc
    # directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\python_experimenting"
    directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear\\python_experimenting"
    (train_model_params, train_solutions, test_model_params, test_solutions) = read_datasets(directory)
    num_dofs = train_solutions.shape[2]
    num_model_params = train_model_params.shape[1]
    num_train_samples = train_solutions.shape[0]
    if (train_model_params.shape[0] !=num_train_samples):
        raise Exception("The number of training samples must be the same in the model parameters and the solutions datasets")

    # Train (or read from disc) the networks
    (encoder_model, decoder_model, loss_cae) = create_cae(num_train_samples, num_dofs, latent_space_dim, train_solutions, load_cae, save_cae, directory)
    (ffnn_model, loss_ffnn) = create_ffnn(num_train_samples, num_model_params, latent_space_dim, train_model_params, train_solutions,
                encoder_model, load_ffnn, save_ffnn, directory)

    # Test surrogate
    print('_________________________________________________________________')
    print('Testing models')
    mean_error_cae = 0
    (mean_error_cae, min_error_cae, max_error_cae) = test_cae(encoder_model, decoder_model, test_solutions)
    mean_error_surrogate = 0
    mean_error_surrogate = test_full_surrogate(decoder_model, ffnn_model, test_model_params, test_solutions)


    # Print results
    print('_________________________________________________________________')
    if len(loss_cae) > 0:
        print("CAE loss function: at start = " + str(loss_cae[0]) + " - at end = " + str(loss_cae[-1]))
    if len(loss_ffnn) > 0:
        print("FFNN loss function: at start = " + str(loss_ffnn[0]) + " - at end = " + str(loss_ffnn[-1]))
    if mean_error_cae > 0:
        print('CAE error on test set: mean = ' + str(mean_error_cae) + ' - min = ' + str(min_error_cae) + ' - max = ' + str(max_error_cae))
    if mean_error_surrogate > 0:
        print('CAE-FFNN surrogate mean error on test set (|expected - predicted| / |expected| = ' + str(mean_error_surrogate))

