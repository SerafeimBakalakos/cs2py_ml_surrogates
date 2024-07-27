import pathlib

import numpy as np
from numpy._typing import NDArray

# import tensorflow as tf


def load_array2D(path: str, float_type):
    """
    :param path: Full path of a text file containing the data
    :param float_type: Floating point type for numpy arrays. Valid values: numpy.single or numpy.double
    :return: An (m, n) numpy array
    """
    extension = pathlib.Path(path).suffix
    if extension == '.txt':
        x = np.loadtxt(path, dtype=float_type, delimiter=' ')
    elif extension == '.npy':
        x = np.load(path)
    else:
        raise Exception('File extension must be .txt or .npy')
    return x


def load_row_matrix(path: str, float_type):
    """
    :param path: Full path of a text file containing the data
    :param float_type: Floating point type for numpy arrays. Valid values: numpy.single or numpy.double
    :return: An (1, n) array, where n = number of entries in input vector
    """
    extension = pathlib.Path(path).suffix
    # extension = path[-3:].lower()
    if extension == '.txt':
        x = np.loadtxt(path, dtype=float_type, delimiter=' ')
    elif extension == '.npy':
        x = np.load(path)
    else:
        raise Exception('File extension must be .txt or .npy')
    return np.expand_dims(x, axis=0)


def save_row_matrix(y: NDArray, path: str):
    """
    :param y: An (1, n) array, where n = number of entries in output vector
    :param path: Full path of a text file containing the data
    """
    extension = pathlib.Path(path).suffix
    if extension == '.txt':
        np.savetxt(path, y, delimiter=' ', newline=' ', fmt='%G')
    elif extension == '.npy':
        y = np.squeeze(y, axis=0)
        np.save(path, y)
    else:
        raise Exception('File extension must be .txt or .npy')


def squeeze_and_save_tensor(y: NDArray, path: str):
    """
    :param y: A vector represented as a (1, 1, ..., n, 1, ..., 1 ) TensorFlow tensor,
                where n = number of entries of the intended output vector
    :param path: Full path of a text file containing the data
    """
    extension = pathlib.Path(path).suffix
    y = np.squeeze(y)
    if extension == '.txt':
        np.savetxt(path, y, delimiter=' ', newline=' ', fmt='%G')
    elif extension == '.npy':
        np.save(path, y)
    else:
        raise Exception('File extension must be .txt or .npy')