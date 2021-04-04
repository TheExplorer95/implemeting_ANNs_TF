from params import *
import tensorflow as tf


def get_classifier(c_dim, num_classes, reduce_model):
    """
    Linear classifier with dimensionality reduction layer at begin and softmax layer at the end.
    :param c_dim: int, dimension of embeddings
    :param num_classes: int, num. different classes that are one-hot encoded
    :param reduce_model: tf.keras.Model, model used to reduce dimensionality of embeddings
    :return: tf.keras.Model
    """
    embedding_inputs = tf.keras.Input(shape=(c_dim))
    x = reduce_model.get_embeddings(
        embedding_inputs
    )  # reduce dimensionality by using reduce_model
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(
        inputs=embedding_inputs, outputs=outputs, name="music_classifier"
    )
    return model


class DimensionalityReduction(tf.keras.Model):
    def __init__(self, c_dim, r_dim):
        """
        Autoencoder used to reduce dimensionality of generated embeddings.
        :param c_dim: int, dimension of context vector (embeddings)
        :param r_dim: int, dimension to reduce down to
        """
        super(DimensionalityReduction, self).__init__()
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.activ = tf.nn.leaky_relu

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(256, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(128, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(64, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.r_dim, activation=self.activ),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(64, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(128, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(256, activation=self.activ),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.c_dim, activation=self.activ),
            ]
        )

    def call(self, x, training):
        x = self.encoder(x, training)
        x = self.decoder(x, training)

        return x

    def get_embeddings(self, x):
        return self.encoder(x)
