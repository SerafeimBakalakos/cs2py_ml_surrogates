import json
import string
import sys

import numpy as np
from numpy._typing import NDArray


def load_input(path:string, np_dtype):
    extension = path[-3:].lower()
    if extension == 'txt':
        x = np.loadtxt(path, dtype=np_dtype, delimiter=' ')
    elif extension == 'npy':
        x = np.load(path)
    else:
        raise Exception('File extension must be .txt or .npy')
    return np.expand_dims(x, axis=0)


def save_output(y:NDArray, path:string):
    extension = path[-3:].lower()
    if extension == 'txt':
        np.savetxt(path, y, delimiter=' ', newline=' ', fmt='%G')
    elif extension == 'npy':
        y = np.squeeze(y, axis=0)
        np.save(path, y)
    else:
        raise Exception('File extension must be .txt or .npy')


if __name__ == '__main__':

    try:
        path_settings = sys.argv[1]
        path_in = sys.argv[2]
        path_out = sys.argv[3]

        #path_settings = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\eval\\settings.json"
        #path_in = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\eval\\x.txt"
        #path_out = "C:\\Users\\Serafeim\\Desktop\\AISolve\\PythonCSharpBridge\\eval\\y.txt"

        settings = None
        with open(path_settings) as file_set:
            settings = json.load(file_set)

        float_type = np.double if settings["Float64"] is True else np.single
        x = load_input(path_in, float_type)
        y = 2.0 * x
        save_output(y, path_out)
    except:
        sys.exit(100)
    else:
        sys.exit(0)

