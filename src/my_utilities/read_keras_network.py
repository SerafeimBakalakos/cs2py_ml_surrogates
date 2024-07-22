import tensorflow as tf

def read_keras_layer(json_layer: dict):
    name = json_layer["Name"]
    if (name == "InputLayer"):
        return tf.keras.layers.InputLayer(input_shape=json_layer["InputShape"])
    elif (name == "Flatten"):
        return tf.keras.layers.Flatten()
    elif (name == "Reshape"):
        return tf.keras.layers.Reshape(target_shape=json_layer["TargetShape"])
    elif (name == "LeakyReLU"):
        return tf.keras.layers.LeakyReLU()
    elif (name == "Dense"):
        return tf.keras.layers.Dense(units=json_layer["Units"])
    elif (name == "Conv1D"):
        return tf.keras.layers.Conv1D(filters=json_layer["Filters"], kernel_size=json_layer["KernelSize"],
                                      strides=json_layer["Strides"], padding=json_layer["Padding"])
    elif (name == "Conv1DTranspose"):
        return tf.keras.layers.Conv1DTranspose(filters=json_layer["Filters"], kernel_size=json_layer["KernelSize"],
                                               strides=json_layer["Strides"], padding=json_layer["Padding"])
    else:
        raise NotImplementedError("This layer is not supported yet")


def create_model_sequential(json_layers_list: list):
    model = tf.keras.Sequential()
    for layer in json_layers_list:
        model.add(read_keras_layer(layer))
    return model
