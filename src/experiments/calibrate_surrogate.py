import numpy as np
import tensorflow as tf

from src.my_utilities import arrayIO

# Returns a tuple (encoder_model, decoder_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_cae(num_dofs:int, latent_space_dim:int, train_solutions, load_cae:bool, save_cae:bool, directory:str):
    path_encoder = directory + "\\encoder_model.keras"
    path_decoder = directory + "\\decoder_model.keras"

    if (load_cae):
        print("\nReading CAE models from disc")
        encoder_model = tf.keras.models.load_model(path_encoder)
        decoder_model = tf.keras.models.load_model(path_decoder)
        lst_loss_history = []
        return (encoder_model, decoder_model, lst_loss_history)
    else:
        (encoder_model, decoder_model, lst_loss_history) = train_cae(num_dofs, latent_space_dim, train_solutions)
        if (save_cae):
            encoder_model.save(path_encoder)
            decoder_model.save(path_decoder)
        return (encoder_model, decoder_model, lst_loss_history)


# Returns the tuple (ffnn_model, lst_loss_history) of keras models and the history of loss function evaluations per epoch
def create_ffnn(num_model_params:int, latent_space_dim:int, train_model_params, train_solutions, encoder_model,
                load_ffnn:bool, save_ffnn:bool, directory:str):
    path_ffnn = directory + "\\ffnn_model.keras"

    if (load_ffnn):
        print("\nReading FFNN model from disc")
        ffnn_model = tf.keras.models.load_model(path_ffnn)
        lst_loss_history = []
        return (ffnn_model, lst_loss_history)
    else:
        (ffnn_model, lst_loss_history) = train_ffnn(num_model_params, latent_space_dim, train_model_params, train_solutions, encoder_model)
        if (save_ffnn):
            ffnn_model.save(path_ffnn, lst_loss_history)
        return (ffnn_model, lst_loss_history)

def train_cae(num_dofs:int, latent_space_dim:int, train_solutions):
    # Training properties
    cae_batch_size = 20
    cae_num_epochs = 100
    cae_learning_rate = 5E-4

    # Network architecture
    cae_encoder = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_space_dim)
    ])

    cae_decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=latent_space_dim),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape(target_shape=(1, 32)),
        tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv1DTranspose(filters=num_dofs, kernel_size=5, strides=1, padding='same')
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
    history = cae_model.fit(train_solutions, train_solutions, batch_size=cae_batch_size, epochs=cae_num_epochs, shuffle=True)
    # cae_model.summary()
    print("_________________________________________________________________")
    return (cae_encoder, cae_decoder, history.history['loss'])



def train_ffnn(num_model_params:int, latent_space_dim:int, train_model_params, train_solutions, encoder_model):
    # Training properties
    ffnn_batch_size = 20
    ffnn_num_epochs = 3000
    ffnn_learning_rate = 1E-4
    ffnn_hidden_size = 64

    # Architecture
    ffnn_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=num_model_params),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(ffnn_hidden_size),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(latent_space_dim)
    ])

    # Compile
    ffnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ffnn_learning_rate), loss='mse')
    ffnn_output = encoder_model(train_solutions)

    # Fit
    print("\nTraining FFNN")
    history = ffnn_model.fit(train_model_params, ffnn_output,
                             batch_size=ffnn_batch_size, epochs=ffnn_num_epochs, shuffle=True)
    print("_________________________________________________________________")

    return (ffnn_model, history.history['loss'])


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
    directory = "C:\\Users\\Serafeim\\Desktop\\AISolve\\CantileverDynamicLinear\\python_experimenting"
    # directory = "C:\\Users\\cluster\\Desktop\\Serafeim\\results\\CantileverDynamicLinear\\python_experimenting"
    (train_model_params, train_solutions, test_model_params, test_solutions) = read_datasets(directory)
    num_dofs = train_solutions.shape[2]
    num_model_params = train_model_params.shape[1]

    # Train (or read from disc) the networks
    (encoder_model, decoder_model, loss_cae) = create_cae(num_dofs, latent_space_dim, train_solutions, load_cae, save_cae, directory)
    (ffnn_model, loss_ffnn) = create_ffnn(num_model_params, latent_space_dim, train_model_params, train_solutions,
                encoder_model, load_ffnn, save_ffnn, directory)

    # Test surrogate
    mean_error = test_full_surrogate(decoder_model, ffnn_model, test_model_params, test_solutions)

    # Print results
    print('_________________________________________________________________')
    if len(loss_cae) > 0:
        print("CAE loss function: at start = " + str(loss_cae[0]) + " - at end = " + str(loss_cae[-1]))
    if len(loss_ffnn) > 0:
        print("FFNN loss function: at start = " + str(loss_ffnn[0]) + " - at end = " + str(loss_ffnn[-1]))
    print('CAE-FFNN surrogate mean error on test seat (|expected - predicted| / |expected| = ' + str(mean_error))

