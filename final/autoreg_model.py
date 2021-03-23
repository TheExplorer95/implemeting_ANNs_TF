import tensorflow as tf

class Autoregressive(tf.keras.Model):
    '''
    g_ar: GRU RNN
    '''

    def __init__ (self, c_dim):
        super(Autoregressive, self).__init__()
        # input dim: [batch, T, z]
        self.gru = tf.keras.layers.GRU(c_dim, name='ar_context') 
        # output dim:[batch, c] since return_seq is False
    @tf.function
    def call (self, z):
        return self.gru(z)