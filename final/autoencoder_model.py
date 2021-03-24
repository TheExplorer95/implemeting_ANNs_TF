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
