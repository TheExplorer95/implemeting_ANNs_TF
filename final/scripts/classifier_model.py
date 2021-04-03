import tensorflow as tf
import numpy as np


def get_classifier(c_dim, num_classes, reduce_model):
    x = tf.keras.Input(shape=(c_dim))
    x = reduce_model.get_embeddings(x)  # first reduce dim
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(
        inputs=embedding_inputs, outputs=outputs, name="music_classifier"
    )
    return model


class DimensionalityReduction(tf.keras.layers.Layer):
    def __init__(self, c_dim, r_dim):
        super(DimensionalityReduction, self).__init__()
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.activ = 'leaky_relu'
        self.layers = np.array([tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(self.r_dim, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(self.c_dim, activation=self.activ)])

    def call(self, x, training):
        for l in self.layers:
            try:
                x = l(x, training)
            except:
                x = l(x)
        return x

    def get_embeddings(self, x):
        for l in self.layers[0, 2, 4]:
            x = l(x)
        return x
