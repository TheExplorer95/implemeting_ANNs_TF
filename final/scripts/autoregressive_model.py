import numpy as np
import tensorflow as tf


class GRU_AR(tf.keras.layers.Layer):
    """
    GRU RNN that takes a sequence of audio window embeddings and combines them into a context embedding.
    c_dim: length of context embedding vector
    """

    def __init__(self, c_dim):
        super(GRU_AR, self).__init__()
        self.gru = tf.keras.layers.GRU(
            c_dim,
            name="ar_context",
        )

    def call(self, z_sequence):
        # input dim: [batch, T, z]
        return self.gru(z_sequence)  # output dim:[batch, c]


#### TRANSFORMER LAYER CLASS AND FUNCTIONS from tensorflow tutorial
# https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)  # (1, position, d_model)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class Trans_EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(Trans_EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class Transformer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_enc_layers,
        num_heads,
        z_dim,
        dff,
        dense_units,
        activation,
        maximum_position_encoding,
        rate=0.1,
    ):
        """
        num_enc_layers: num. transformer encoder layers to be stacked
        num_heads: num. sets of q,k,v
        z_dim: z_dim
        dff: num. units for first dense layer within encoder layer
        dense_units: list of num. units for additional dense layers, last number is c_dim
        activation: activation func. to use for additional dense layers
        maximum_position_encoding: T in our case, max length of sequence
        rate: dropout rate
        """
        super(Transformer, self).__init__()

        self.z_dim = z_dim
        self.num_enc_layers = num_enc_layers

        # embedding layer isn't needed as the input is already embedded
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, z_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.z_dim)

        self.enc_layers = [
            Trans_EncoderLayer(z_dim, num_heads, dff, rate)
            for _ in range(num_enc_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

        # additional dense layer and dropouts at the end
        self.fcnet = [
            [tf.keras.layers.Dense(n_l, activation), tf.keras.layers.Dropout(rate)]
            for n_l in dense_units
        ]
        self.fcnet = [l for sublist in self.fcnet for l in sublist]  # flatten
        del self.fcnet[-1]  # last dropout

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]  # T+kN

        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, z_dim)
        x *= tf.math.sqrt(tf.cast(self.z_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_enc_layers):
            x = self.enc_layers[i](
                x, training, mask
            )  # (batch_size, input_seq_len, z_dim)

        x = tf.keras.layers.Flatten()(x)  # (batch_size, input_seq_len*z_dim)
        for i in range(len(self.fcnet)):
            if i % 2 == 0:
                x = self.fcnet[i](x)  # dense layer
            else:
                x = self.fcnet[i](x, training)  # dropout

        return x  # (batch_size, c_dim)
