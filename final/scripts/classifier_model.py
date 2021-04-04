from params import *
import tensorflow as tf
import numpy as np


def get_classifier(c_dim, num_classes, reduce_model):
    embedding_inputs = tf.keras.Input(shape=(c_dim))

    ### TODO: set trainable = False
    x = reduce_model.get_embeddings(embedding_inputs)  # first reduce dim
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(
        inputs=embedding_inputs, outputs=outputs, name="music_classifier"
    )
    model.summary()
    model.layers[0].trainable = False # do not train autoencoder
    return model



class DimensionalityReduction(tf.keras.Model):
    def __init__(self, c_dim, r_dim):
        super(DimensionalityReduction, self).__init__()
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.activ = tf.nn.leaky_relu

        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=self.activ),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(64, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(self.r_dim, activation=self.activ)

        ])
        self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation=self.activ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(self.c_dim, activation=self.activ),
        ])

    def call(self, x, training):
        x= self.encoder(x,training)
        x= self.decoder(x, training)

        return x

    def get_embeddings(self, x):
        return self.encoder(x)
