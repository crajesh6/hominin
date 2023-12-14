import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import tensor_shape
import tensorflow_probability as tfp


class PairwiseConv1D(keras.layers.Conv1D):
    """
    This class implement pairwise convolutions for 1D signals.  Standard
    convolutions implemented as `keras.layers.Conv1D` perform linear
    transformations of patches from the input signal. Pairwise convolutions
    perform linear transformations of all pairwise terms from entries in
    all patches of the input signal. The implementation is achieved by
    taking an outer product of each patch and performing a linear
    transformation of the pairwise (i.e. lower diagonal) terms. The rest of
    this docstring is copied from the keras.layers.Conv1D docstring.
    """
    __doc__ = __doc__ + super.__doc__
    padding_map_dict = {'same':'SAME', 'valid':'VALID'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_patches(self, x):
        """
        x -> (N, L, A)
        This module is tested for correctness.
        """
        assert len(x.shape) == 3

        x = tf.expand_dims(x, axis=2) ## (N, L, 1, A)
        sizes = [1, self.kernel_size[0], 1, 1]
        strides = [1, self.strides[0], 1, 1]
        rates = [1, self.dilation_rate[0], 1, 1]
        padding = self.padding_map_dict[self.padding]
        xp = tf.image.extract_patches(x,sizes=sizes,strides=strides,rates=rates,padding=padding)
        xp = tf.squeeze(xp, axis=2)  ## (N, num patches, flattened patch size)
        return xp

    def _outer_product(self, xpatches):
        """
        xpatches -> (N, numpatches, patch size*A)
        RETURNS:
        xout -> (N, num patches, patch size*A, patch_size*A)
        """
        res = tf.einsum("ijk, ijl -> ijkl", xpatches, xpatches)  ## (N, numpatches, P*A, P*A)
        res = tf.linalg.set_diag(res, diagonal=xpatches) ## replace the sq. term with unit power in the diag.
        return res

    @property
    def full_kernel(self):
        k = tf.transpose(self.kernel, [1,0])  ## (C, numweights,)
        k = tfp.math.fill_triangular(k) ## (C, P*A, P*A)
        k = tf.transpose(k, [1, 2, 0])  ## (P*A, P*A, C)
        return k

    @property
    def diag_kernel(self):
        """
        Returns the diagonal of the kernel.
        """
        k = self.full_kernel  ## (P*A, P*A, C)
        k = tf.transpose(k, [2, 0, 1])  ## (C, P*A, P*A)
        k = tf.linalg.diag_part(k) ## (C, P*A)
        k = tf.transpose(k, [1, 0]) ## (P*A, C)
        return k

    def build(self, input_shape):
        """
        Expected input_shape is (N, L, A)
        """
        input_shape = tensor_shape.TensorShape(input_shape)  #(L, A)
        A = input_shape[-1]  # A
        P = self.kernel_size[0] # P
        flat_patch_size = P*A
        kernel_shape = [ int(flat_patch_size*(flat_patch_size+1)*0.5), self.filters ] ## (numweights, C)

        # add the kernel
        self.kernel = self.add_weight(
                    name='kernel',
                    shape=kernel_shape,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True,
                    dtype=self.dtype,
                        )

        # add the bias
        if self.use_bias:
            self.bias = self.add_weight(
                      name = 'bias',
                      shape = (self.filters,),
                      initializer=self.bias_initializer,
                      regularizer=self.bias_regularizer,
                      constraint=self.bias_constraint,
                      trainable=True,
                      dtype=self.dtype
                          )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        inputs -> (N, L, in_channels)
        RETURNS:
        outputs -> (N, L, out_channels)
        """

        xp = self._get_patches(inputs) ## (N, numpatches, P*A)

        # take the outer product
        xout = self._outer_product(xp) ## (N, numpatches, P*A, P*A)

        # compute the output
        kern = self.full_kernel  ## (P*A, P*A, C)
        outputs = tf.einsum("ijkl, klm -> ijm",xout, kern)

        # add the bias
        if self.use_bias:
            outputs = outputs + self.bias

        # apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class PairwiseReg(keras.regularizers.Regularizer):
    """

    A regularizer than applies separate regularization functions on
    the diagonal and off-diagonal terms in the pairwise kernel.
    """
    def __init__(self, diag, offdiag, *args, **kwargs):
        """
        diag_regularizer <keras.regularizer.Regularizer> - The
        offdiag_regularizer <keras.regularizer.Regularizer>
        """
        super().__init__(*args, **kwargs)
        self.diag = diag
        self.offdiag = offdiag

    def __call__(self, x):
        """
        x -> Pairwise kernel (expected shape = (numterms, A, C))
        """
        ndims = len(x.shape)
        perm = list(np.arange(1, ndims)) + [0]
        x = tf.transpose(x, perm)  ## move 1st dimension to the end
        x = tfp.math.fill_triangular(x)

        diag_part = tf.linalg.diag_part(x)  ##(A, C, P)
        offdiag_part = x - tf.linalg.diag(diag_part) ## (A, C, P, P)

        res1 =  self.diag(diag_part)
        res2 = self.offdiag(offdiag_part)
        return res1+res2

    def get_config(self):
        config = {}
        config['diag'] = self.diag
        config['offdiag'] = self.offdiag
        return config



class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, embedding_size=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size == None else embedding_size

        assert d_model % self.num_heads == 0 and d_model % 6 == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model, use_bias=False)
        self.wk = keras.layers.Dense(d_model, use_bias=False)
        self.wv = keras.layers.Dense(d_model, use_bias=False)

        self.r_k_layer = keras.layers.Dense(d_model, use_bias=False)
        self.r_w = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True)
        self.r_r = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)
        q = q / tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32))

        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size=self.embedding_size//6

        seq_length = tf.cast(seq_len, dtype=tf.float32)
        exp1 = f_exponential(tf.abs(pos), feature_size, seq_length=seq_length)
        exp2 = tf.multiply(exp1, tf.sign(pos)[..., tf.newaxis])
        cm1 = f_central_mask(tf.abs(pos), feature_size, seq_length=seq_length)
        cm2 = tf.multiply(cm1, tf.sign(pos)[..., tf.newaxis])
        gam1 = f_gamma(tf.abs(pos), feature_size, seq_length=seq_length)
        gam2 = tf.multiply(gam1, tf.sign(pos)[..., tf.newaxis])

        # [1, 2seq_len - 1, embedding_size]
        positional_encodings = tf.concat([exp1, exp2, cm1, cm2, gam1, gam2], axis=-1)
        positional_encodings = keras.layers.Dropout(0.1)(positional_encodings)

        # [1, 2seq_len - 1, d_model]
        r_k = self.r_k_layer(positional_encodings)

        # [1, 2seq_len - 1, num_heads, depth]
        r_k = tf.reshape(r_k, [r_k.shape[0], r_k.shape[1], self.num_heads, self.depth])
        r_k = tf.transpose(r_k, perm=[0, 2, 1, 3])
        # [1, num_heads, 2seq_len - 1, depth]

        # [batch_size, num_heads, seq_len, seq_len]
        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)

        # [batch_size, num_heads, seq_len, 2seq_len - 1]
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        # [batch_size, num_heads, seq_len, seq_len]
        relative_logits = relative_shift(relative_logits)

        # [batch_size, num_heads, seq_len, seq_len]
        logits = content_logits + relative_logits
        attention_map = tf.nn.softmax(logits)

        # [batch_size, num_heads, seq_len, depth]
        attended_values = tf.matmul(attention_map, v)
        # [batch_size, seq_len, num_heads, depth]
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attended_values, [batch_size, seq_len, self.d_model])

        output = self.dense(concat_attention)

        return output, attention_map



#------------------------------------------------------------------------------------------
# Positional encoding functions for Multi-Head Attention
#------------------------------------------------------------------------------------------


def f_exponential(positions, feature_size, seq_length=None, min_half_life=3.0):
    if seq_length is None:
        seq_length = tf.cast(tf.reduce_max(tf.abs(positions)) + 1, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = tf.reshape(half_life, shape=[1]*positions.shape.rank + half_life.shape)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    return outputs

def f_central_mask(positions, feature_size, seq_length=None):
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32)) - 1
    center_widths = tf.reshape(center_widths, shape=[1]*positions.shape.rank + center_widths.shape)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    return outputs

def f_gamma(positions, feature_size, seq_length=None):
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    stdv = seq_length / (2*feature_size)
    start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = tf.reshape(mean, shape=[1]*positions.shape.rank + mean.shape)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2
    def gamma_pdf(x, conc, rt):
        log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
        log_normalization = (tf.math.lgamma(concentration) - concentration * tf.math.log(rate))
        return tf.exp(log_unnormalized_prob - log_normalization)
    probabilities = gamma_pdf(tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis], concentration, rate)
    outputs = probabilities / tf.reduce_max(probabilities)
    return outputs

def relative_shift(x):
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x
