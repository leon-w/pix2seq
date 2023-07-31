import keras_core as keras


def get_variable_initializer(scale=1e-10):
    return keras.initializers.VarianceScaling(
        scale=scale,
        mode="fan_avg",
        distribution="uniform",
    )
