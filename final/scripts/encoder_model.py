import tensorflow as tf


class Conv1DEncoder(tf.keras.layers.Layer):
    """
    Encodes an input 1D sequence into an audio window embedding.
    z_dim: size of embedding
    stride_sizes: list of stride arguments for Conv1D layers
    kernel_sizes: list of kernel size arguments for Conv1D layers
    n_filters:    list of filter number arguments for Conv1D layers
    activation:   activation function used in Conv1D layers and for output Dense layer. (e.g. "relu" or tf.nn.relu)
    """

    def __init__(self, z_dim, stride_sizes, kernel_sizes, n_filters, activation):
        super(Conv1DEncoder, self).__init__()

        s = stride_sizes
        k = kernel_sizes
        f = n_filters

        self.enc_layers = []

        for l in range(len(f)):
            self.enc_layers.append(tf.keras.layers.Conv1D(f[l], k[l], s[l]))
            self.enc_layers.append(tf.keras.layers.BatchNormalization())
            self.enc_layers.append(tf.keras.layers.Activation(activation))
        self.enc_layers.append(tf.keras.layers.Flatten())
        self.enc_layers.append(tf.keras.layers.Dropout(0.1))
        self.enc_layers.append(tf.keras.layers.Dense(512))
        self.enc_layers.append(tf.keras.layers.Activation(activation))
        self.enc_layers.append(tf.keras.layers.Dropout(0.1))
        self.enc_layers.append(tf.keras.layers.Dense(z_dim))
        self.enc_layers.append(tf.keras.layers.Activation(activation))

    def call(self, x, training):
        # input dim: [batch, window_size, 1]
        for l in self.enc_layers:
            try:
                x = l(x, training)
            except:
                x = l(x)

        # ouput dim:[batch, z_dim]
        return x


class Conv2DEncoder(tf.keras.layers.Layer):
    """
    g_enc: strided 2d convolution
    """

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
            self.enc_layers.append(tf.keras.layers.BatchNormalization())
            self.enc_layers.append(tf.keras.layers.Activation(conv_fct))

        self.enc_layers.append(tf.keras.layers.GlobalMaxPooling2D())

        for l in range(len(d)):
            self.enc_layers.append(tf.keras.layers.Dense(units=d[l]))
            self.enc_layers.append(tf.keras.layers.Activation(dense_act))
            # self.enc_layers.append(tf.keras.layers.Dropout(0.1))

        self.enc_layers.append(tf.keras.layers.Dense(self.z_dim))
        self.enc_layers.append(tf.keras.layers.Activation(dense_act))

    def call(self, x, training):
        for l in self.enc_layers:
            try:  # batch normalization
                x = l(x, training)
            except Exception:
                x = l(x)

        return x
