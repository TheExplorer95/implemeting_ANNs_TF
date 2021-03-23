import tensorflow as tf

class Encoder(tf.keras.Model):
    '''
    g_enc: strided 1d convolution
    '''

    def __init__ (self, z_dim):
        super(Encoder, self).__init__()
        s = [8,4,2,2,2]  # stride sizes
        k = [20,10,4,4,4]  # kernel sizes
        f = [128,512,256,128,128]  # num filters

        # input dim: [batch, T+K*N, d, 1]
        self.enc_layers = []
        for l in range(5):
            self.enc_layers.append(tf.keras.layers.Conv1D(f[l],k[l],s[l]))
            self.enc_layers.append(tf.keras.layers.BatchNormalization())
            self.enc_layers.append(tf.keras.layers.LeakyReLU())
        self.enc_layers.append(tf.keras.layers.GlobalAveragePooling1D())
        self.enc_layers.append(tf.keras.layers.Dense(z_dim, activation='tanh'))
        # ouput dim:[batch, T+K*N, z]
    @tf.function
    def call (self, x, training):

        for l in self.enc_layers:
            try:  # batch normalization
                x = l(x, training)
            except:
                x = l(x)
        return x

class CNN_Encoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, latent_dim):
        super(CNN_Encoder, self).__init__()

        self.layers = []

        # (2,2) strided convolution to downsample
        # padding=same for padding of the input image

        for i in range(4):
            self.layers.append(tf.keras.layers.Conv2D(filters=32,
                                                      kernel_size=(3,3),
                                                      strides=(2,2),
                                                      padding='same',
                                                      input_shape=input_dim))
            self.layers.append(tf.keras.layers.BatchNormalization())
            self.layers.append(tf.keras.layers.Activation('relu'))

        self.layers.append(tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(3,3),
                                                  strides=(2,2),
                                                  padding='same'))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation('relu'))

        # output layer (latent_space)
        self.layers.append(tf.keras.layers.GlobalAveragePooling1D())
        self.layers.append(tf.keras.layers.Dense(latent_dim, activation='tanh'))

    def call(self, x, training=False):
        for layer in self.layers:
            try:  # training argument only for BN layer
                x = layer(x, training)
            except:
                x = layer(x)
        return x


class CNN_Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, output_dim, restore_shape):
        super(CNN_Decoder, self).__init__()
        self.layers = []

        # dense layer to restore dim of flattend data
        self.layers.append(tf.keras.layers.Dense(units=int(tf.math.reduce_prod((restore_shape))),
                                                 input_shape=(latent_dim,)))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation('relu'))

        # reshape to 3 dim with depth dim again
        self.layers.append(tf.keras.layers.Reshape(target_shape=restore_shape))

        # (2,2) strided transposed conv to upsample

        self.layers.append(tf.keras.layers.Conv2DTranspose(filters=32,
                                                           kernel_size=(3,3),
                                                           strides=(2,2),
                                                           padding='same'))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation('relu'))

        # restore image by convolution with image size
        self.layers.append(tf.keras.layers.Conv2DTranspose(filters=1,
                                                  kernel_size=(3,3),
                                                  strides=(2,2),
                                                  padding='same'))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation('sigmoid'))

    def call(self, x, training=False):
        for layer in self.layers:
            try:  # training argument only for BN layer
                x = layer(x, training)
            except:
                x = layer(x)
        return x

class CNN_Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, restore_shape=(7,7,64)):
        super(CNN_Autoencoder, self).__init__()
        # encoder and decoder are symmetric
        self.encoder = CNN_Encoder(input_dim=input_dim,
                                   latent_dim=latent_dim)
        self.decoder = CNN_Decoder(latent_dim=latent_dim,
                                   output_dim=input_dim,
                                   restore_shape=restore_shape)

    def call(self, x, training=False):
        x = self.encoder(x, training)
        self.latent_repr = x  # keep latent_repr as property in case it should be analyzed
        x = self.decoder(x, training)
        return x
