import tensorflow as tf
from autoreg_model import Autoregressive
from autoencoder_model import Encoder

if __name__=='__main__':
    en = Encoder(128)
    print("done")

#@tf.function
def compute_f (z, z_pred):
    '''
    compute f following eq(3) in the paper to be batch (K x N) matrices.
    First column is the postive sample.
    '''

    # z input dim: [batch, K, N, z],
    z = tf.expand_dims(z, axis=-2)  # [batch, K, N, 1, z]

    # z_pred input dim: [batch, K, z]
    pred = tf.repeat(z_pred, repeats=z.shape[2], axis=-2)  # [batch, K*N, z]
    pred = tf.reshape(pred, shape=[z.shape[0],z.shape[1],z.shape[2],z.shape[-1]])  # [batch, K, N, z]
    pred = tf.expand_dims(pred, axis=-1)  # [batch, K, N, z, 1]

    dot_prod = tf.linalg.matmul(z, pred)  # [batch, K, N, 1, 1]
    dot_prod = tf.squeeze(dot_prod, axis=[-2,-1])  # [batch, K, N]
    dot_prod = tf.exp(dot_prod)
    return dot_prod  # output dim: [batch, K, N]

class Predict_z (tf.keras.layers.Layer):
    '''
    transformation of c_t, currently linear (W_k) for all future timesteps
    '''

    def __init__ (self, z_dim, K):
        super(Predict_z, self).__init__()

        # input_dim: [batch, c]
        self.transform_layers = []
        for k in range(K):  # k different layers for each timestep
            self.transform_layers.append(tf.keras.layers.Dense(z_dim))
    #@tf.function
    def call(self, c_t):
        # TODO: maybe size should be multidimensional
        z_pred = tf.TensorArray(tf.float32, size=len(self.transform_layers))
        for l in tf.range(len(self.transform_layers)):
            z_pred = z_pred.write(l, self.transform_layers[l](c_t))  # apply for each k
            z_pred_t = z_pred.stack()
            # [K, batch, z]
        return tf.transpose(z_pred_t, perm=[1,0,2])  # output_dim: [batch, K, z]

class CPC (tf.keras.models.Model):
    '''
    put everything together. Return f_k for every k
    '''

    def __init__ (self, num_time_observations, num_time_future, num_negative_samples, z_dim, c_dim):
        super(CPC, self).__init__()
        self.T = num_time_observations
        self.K = num_time_future
        self.N = num_negative_samples
        self.z = z_dim
        self.c = c_dim

        self.g_enc = Encoder(self.z)
        self.g_ar = Autoregressive(self.c)
        self.p_z = Predict_z(z_dim=self.z, K=self.K)
    #@tf.function
    def call(self, x, training=False):
        # input dim: [batch, T+K*N, d, 1]
        #print('input dim: ', x.shape)
        # Embedding
        z_t = tf.keras.layers.TimeDistributed( # dim 1 is the temporal dim
            self.g_enc)(x, training=training)  # [batch, T+K*N, z]
        #print('embedding dim: ', z_t.shape)


        # Split current observation embeddings and future embeddings
        z_obs = z_t[:, :self.T]  # t = {0,...,T}, dim: [batch, T, z]
        z_future = z_t[:, self.T:]  # t = {T+1,,,T+K} for N samples, dim:[batch, K*N, z]
        z_future = tf.reshape(z_future, [-1, self.K, self.N, self.z])  # [batch, K, N, z]
        #print('embedding obs:', z_obs.shape)
        #print('embedding pred:', z_future.shape)

        # Predict embeddings
        c_T = self.g_ar(z_obs)  # [batch, c]
        #print('context:', c_T.shape)
        z_pred = self.p_z(c_T)  # [batch, K, z]
        #print('transformed_context:', z_pred.shape)

        # Compute f matrices
        f_mat = compute_f(z_future, z_pred)  # [batch, K, N]

        return f_mat
