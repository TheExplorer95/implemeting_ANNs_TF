from params import *
import tensorflow as tf

from autoregressive_model import GRU_AR, Transformer
from encoder_model import Conv1DEncoder, Conv2DEncoder


class Predict_z(tf.keras.layers.Layer):
    def __init__(self, z_dim, K, mixed_precision=False):
        """
        Generate k different predictions from context vector by linear mappings.
        :param z_dim: int, dimension of latent representation of data
        :param K: int, num. future time steps
        :param mixed_precision: boolean, use mixed_precision to boost performance if True
        """
        super(Predict_z, self).__init__()

        self.transform_layers = []

        if mixed_precision:
            self.z_dtype = tf.float16
        else:
            self.z_dtype = tf.float32

        # one linear layer for each future time-step
        for k in tf.range(K):
            self.transform_layers.append(tf.keras.layers.Dense(z_dim))

    def call(self, c_t):
        # input_dim: [batch, c_dim]
        z_pred = tf.TensorArray(self.z_dtype, size=len(self.transform_layers))

        for l, layer in enumerate(self.transform_layers):
            # apply linear projection layer for each k
            z_pred = z_pred.write(l, layer(c_t))

        z_pred_t = z_pred.stack()
        return tf.transpose(z_pred_t, perm=[1, 0, 2])  # output_dim: [batch, K, z]


def compute_f(z, z_pred):
    """
    Compute f-scores following eq(3) in the paper to be batch (K x N) matrices.
    Computes similarity (f-)scores as the exp of the dot product of two embeddings.
    First column of the returned f-score matrix is the positive sample.
    """
    # z_pred input dim: [batch, K, z]
    # z input dim:      [batch, K, N, z]
    z = tf.expand_dims(z, axis=-2)  # [batch, K, N, 1, z]

    pred = tf.repeat(z_pred, repeats=z.shape[2], axis=-2)  # [batch, K*N, z]
    pred = tf.reshape(
        pred, shape=[z.shape[0], z.shape[1], z.shape[2], z.shape[-1]]
    )  # [batch, K, N, z]
    pred = tf.expand_dims(pred, axis=-1)  # [batch, K, N, z, 1]

    dot_prod = tf.linalg.matmul(z, pred)  # [batch, K, N, 1, 1]
    # cosine_similarity = dot_prod/(tf.norm(z)*tf.norm(pred))  # one could use other similarity scores
    dot_prod = tf.squeeze(dot_prod, axis=[-2, -1])  # [batch, K, N]
    f_mat = tf.exp(dot_prod)
    # output dim: [batch, K, N]
    return f_mat


class CPC(tf.keras.models.Model):
    """


    n_observations:     number of subsequent windows of audio used for prediction
    n_future:           number of future audio windows to predict
    n_negative_samples: number of random negative samples
    z_dim:              audio window encoding size
    encoder_args:       argument dictionary for Encoder model
    """

    def __init__(
        self,
        n_observations,
        n_future,
        n_samples,
        z_dim,
        c_dim,
        encoder,
        autoregressive,
        predict_z,
        encoder_args,
        ar_args,
        m_precision,
    ):
        """
        Full Contrastive Predictive Coding Model.
        :param n_observations: int, num. windows until current time point, T
        :param n_future:  int, num. windows to predict, K
        :param n_samples: int, num. samples, N
        :param z_dim: int, dimension of latent space, output dim of encoder
        :param c_dim: int, dimension of context vector, output dim of autoregressive model
        :param encoder: str, either '1d_conv' or '2d_conv', select which encoder architecture to use
        :param autoregressive: str, either 'GRU' or 'transformer', select which autoregressive architecture to use
        :param predict_z: tf.keras.layer, model used to map context vector to predictions for different k
        :param encoder_args: dict, arguments for encoder
        :param ar_args: dict, arguments for autoregressive model
        :param m_precision: boolean, use mixed precision for performance boost if True
        """

        super(CPC, self).__init__()

        self.T = n_observations
        self.K = n_future
        self.N = n_samples

        self.z = z_dim
        self.c = c_dim

        # encoder
        if encoder == "1d_conv":
            self.g_enc = Conv1DEncoder(**encoder_args)
        elif encoder == "2d_conv":
            self.g_enc = Conv2DEncoder(**encoder_args)

        # autoregressive model
        if autoregressive == "GRU":
            self.g_ar = GRU_AR(self.c)
        elif autoregressive == "transformer":
            self.g_ar = Transformer(**ar_args)

        # prediction model
        self.p_z = predict_z(z_dim=self.z, K=self.K, mixed_precision=m_precision)

    def get_embedding(self, x):
        """
        Generate embeddings (context vector).
        :param x: input data
        :return: context vector of dim (batch_size, c_dim)
        """
        z_t = tf.keras.layers.TimeDistributed(self.g_enc)(x, training=False)
        c_T = self.g_ar(z_t)
        return c_T

    def call(self, x, training=False):
        # input dim: [batch, T+K*N, window_size, 1]
        # Obtain Embeddings for T+k*N time windows of length d
        z_t = tf.keras.layers.TimeDistributed(self.g_enc)(
            x, training=training
        )  # [batch, T+K*N, z_dim]

        # Split into current observation embeddings and (positive and negative) future embeddings
        z_obs = z_t[:, : self.T]  # [batch, T, z]
        z_future = z_t[:, self.T :]  # [batch, K*N, z]
        z_future = tf.reshape(
            z_future, [-1, self.K, self.N, self.z]
        )  # [batch, K, N, z]

        # Obtain context embedding vector for T encoded time-windows
        c_T = self.g_ar(z_obs)  # [batch, c]

        # Linearly project context vector to make predictions of the future encoded time-windows
        z_pred = self.p_z(c_T)  # [batch, K, z]

        # Compute f matrix in which the first column is the f-scores for the positive sample
        f_mat = compute_f(z_future, z_pred)  # output dim: [batch, K, N]

        return f_mat


class InfoNCE(tf.keras.losses.Loss):
    """
    Compute InfoNCE loss given a batch of f matrices with dim (K x N)
    """

    def __init__(self, weighted=False):
        """
        Compute InfoNCE loss given a batch of matrices, (batch_size, K, N)
        :param weighted: use custom weighting factors to average loss over k if True
        """
        self.weighted = weighted  # default is to use uniform averaging

    def __call__(self, f):
        # input dim: [batch, K, N]
        denominator = tf.reduce_sum(f, axis=2)  # -> [batch, K]
        losses = -tf.math.log(
            f[:, :, 0] / denominator
        )  # first column is the positive k predictions
        if self.weighted:
            weights_mask = tf.range(
                1, f.shape[1] + 1, dtype=tf.float32
            )  # e.g. [1,...,k]
            weights_mask = tf.expand_dims(weights_mask, 0)  # e.g. [[1,...,k]]
            weights_mask = tf.reverse(weights_mask, [1])  # e.g. [[k,..,1]]
            weighted_l = tf.math.multiply(weights_mask, losses)
            loss = tf.reduce_mean(weighted_l, axis=None)
            loss = loss / tf.math.reduce_sum(
                tf.range(1, f.shape[1] + 1, dtype=tf.float32)
            )  # normalize with total weight sum
        else:
            loss = tf.reduce_mean(
                losses, axis=None
            )  # Take mean loss over batch_size and K

        return loss
