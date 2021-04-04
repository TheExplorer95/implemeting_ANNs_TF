from params import *
import tensorflow as tf


class Conv1DEncoder(tf.keras.layers.Layer):
    def __init__(self, z_dim, stride_sizes, kernel_sizes, n_filters, activation):
        """
        Encodes 1D sequence (e.g. audio data) into a latent space with dimension of z_dim.
        :param z_dim: int, dimension of latent representation
        :param stride_sizes: list of int, step size for each Conv1D layers
        :param kernel_sizes: list of int, receptive field size for each Conv1D layers
        :param n_filters: list of int, num. different feature maps for each Conv1D layers
        :param activation: activation function used in Conv1D layers and for output Dense layers
        """

        super(Conv1DEncoder, self).__init__()

        s = stride_sizes
        k = kernel_sizes
        f = n_filters

        self.enc_layers = []

        # conv1d layers
        for l in range(len(f)):
            self.enc_layers.append(tf.keras.layers.Conv1D(f[l], k[l], s[l]))
            self.enc_layers.append(tf.keras.layers.BatchNormalization())
            self.enc_layers.append(tf.keras.layers.Activation(activation))

        # dense layers
        self.enc_layers.append(tf.keras.layers.Flatten())
        self.enc_layers.append(tf.keras.layers.Dropout(0.1))
        self.enc_layers.append(tf.keras.layers.Dense(512))
        self.enc_layers.append(tf.keras.layers.Activation(activation))
        self.enc_layers.append(tf.keras.layers.Dropout(0.1))
        self.enc_layers.append(tf.keras.layers.Dense(z_dim))
        self.enc_layers.append(tf.keras.layers.Activation(activation))

    def call(self, x, training):
        # input dim: (batch, window_size, 1)
        for l in self.enc_layers:
            try:
                x = l(x, training)
            except:
                x = l(x)

        # output dim:(batch, z_dim)
        return x


class Conv2DEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        z_dim,
        stride_sizes,
        kernel_sizes,
        n_filters,
        dense_units,
        conv_fct,
        dense_act,
        kernel_reg,
    ):
        """
        Encodes 2D data (e.g. picture) into a latent space with dimension of z_dim.
        :param z_dim: int, dimension of latent representation
        :param stride_sizes: list of int, step size for each Conv2D layer
        :param kernel_sizes: list of int, receptive field size for each Conv2D layer
        :param n_filters: list of int, num. different feature maps for each Conv2D layer
        :param dense_units: list of int, num. of units for each fully connected layer
        :param conv_fct: activation function used for Conv2D layers
        :param dense_act: activation function used for fully connected layers
        :param kernel_reg: boolean, use L2 normalization if true
        """

        super(Conv2DEncoder, self).__init__()

        s = stride_sizes
        k = kernel_sizes
        f = n_filters
        d = dense_units

        self.z_dim = z_dim

        if kernel_reg:
            regularizer = tf.keras.regularizers.l2
        else:
            regularizer = None

        # input dim: [batch, T+K*N, d, 1]
        self.enc_layers = []

        # Conv2D layers
        for l in range(len(s)):
            self.enc_layers.append(
                tf.keras.layers.Conv2D(
                    filters=f[l],
                    kernel_size=k[l],
                    strides=s[l],
                    padding="same",
                    kernel_regularizer=regularizer(),
                )
            )
            self.enc_layers.append(tf.keras.layers.SpatialDropout2D(0.1))
            self.enc_layers.append(tf.keras.layers.Activation(conv_fct))

        # Dense layers
        self.enc_layers.append(tf.keras.layers.Flatten())
        for l in range(len(d)):
            self.enc_layers.append(tf.keras.layers.Dense(units=d[l]))
            self.enc_layers.append(tf.keras.layers.Activation(dense_act))
            self.enc_layers.append(tf.keras.layers.Dropout(0.1))
        self.enc_layers.append(tf.keras.layers.Dense(self.z_dim))
        self.enc_layers.append(tf.keras.layers.Activation(dense_act))

    def call(self, x, training):
        for i, l in enumerate(self.enc_layers):
            try:  # batch normalization
                x = l(x, training)
            except Exception:
                x = l(x)
        return x
