
"""
Defines the models and layers used in the NN.
Includes most layer parameters as well.
"""

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Dropout, GlobalMaxPooling1D, Embedding, TimeDistributed
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.python.ops.gen_array_ops import shape

# default values for the network

embed_dim = 1
# in order NACGT, see nucleotides:
embed_init_w = [0, 1, 0.25, 0.75, 0.5]

sequence_target_length = 100
interpolation = "bilinear"

conv_filters_1 = 12  # 130
units_2 = 204
units_3 = 150
dense_units_1 = 64  # 196
pool_size_1 = 148
pool_size_2 = 236
pool_size_3 = 81
conv_window_length_1 = 31
conv_window_length_2 = 106
conv_window_length_3 = 121
dropout_probability = 0.5

# default value for bias
bias_vector_init_value = 0.1

# the loss is adjusted with a parameter for l2 regularization in each of
# the convolutional layers
penalty_regularization_w_conv = 0.001

# 2 Layer
# network.add(Conv1D(units_2, activation='relu', kernel_size=conv_window_length_2, padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
# network.add(MaxPooling2D(pool_size=pool_size_2, padding='same'))

# 3 Layer
# network.add(Conv1D(units_3, activation='relu', kernel_size=conv_window_length_3, padding='same', use_bias=True, bias_initializer=keras.initializers.Constant(bias_vector_init_value), kernel_regularizer=keras.regularizers.l2(penalty_regularization_w_conv)))
# network.add(MaxPooling2D(pool_size=pool_size_3, padding='same'))


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.bool)])
def masked_lengths(mask):
    """Computes the length of each sequence using 2D mask data; essentially
    the inverse of `tf.sequence_mask`."""
    return tf.shape(mask)[1] - tf.math.argmax(mask[:, ::-1], axis=1, output_type=tf.int32)


class Resizing1D(tf.keras.layers.Layer):
    def __init__(self, length=1000, interpolation="bilinear", name="resizing1d", **kwargs):
        super(Resizing1D, self).__init__(name=name, **kwargs)
        self.resizer = Resizing(1, length, interpolation=interpolation)

    def __resize_with_length(self, t):
        sequence, length = t
        # assumes row shape: batch(1) * height(1) * width * channel:
        tf.debugging.assert_shapes([
            (sequence, (1, 1, "W", "C")),
            (length, (1,))
        ])
        return self.resizer(sequence[:, :, :length, :])

    @tf.function
    def __resize_node(self, input):
        # input shape : node_length * embed_dim
        x = tf.expand_dims(input, axis=0)
        x = tf.expand_dims(x, axis=0)
        tf.debugging.assert_shapes([
            (x, (1, 1, "W", "C"))
        ])
        x = self.resizer(x)
        return tf.squeeze(x, axis=[0, 1])

    def call(self, inputs, mask=None):
        # input shape : num_nodes * node_length * embed_dim
        if isinstance(inputs, tf.RaggedTensor):
            return tf.map_fn(
                self.__resize_node,
                inputs,
                fn_output_signature=tf.TensorSpec(
                    shape=[self.resizer.target_width, None],
                    dtype=tf.float32
                ),
                name="resizing_map_fn"
            )
            # add a dummy height dimension to all sequences:
        x = tf.expand_dims(inputs, axis=1)
        if mask is not None:
            lengths = masked_lengths(mask)
            # add a dummy batch dimension to all sequences:
            x = tf.expand_dims(x, axis=1)
            # output signature is expected to be same as input unless otherwise noted
            x = tf.map_fn(
                self.__resize_with_length,
                (x, lengths),
                fn_output_signature=x.dtype
            )
            # remove dummy dimensions:
            return tf.squeeze(x, axis=[1, 2])
        else:
            return tf.squeeze(self.resizer(x), axis=[1])
            # (remove dummy height dimension)

    def get_config(self):
        config = super(Resizing1D, self).get_config()
        resize_config = self.resizer.get_config()
        config.update({
            "interpolation": resize_config["interpolation"],
            "length": resize_config["width"]
        })
        return config


class CNNModel(tf.keras.Model):

    def __init__(self, voc_size=5, n_classes=2, name=None):
        super(CNNModel, self).__init__(name=name)
        self.n_classes = n_classes
        self.voc_size = voc_size
        # embeds integral indices into dense arrays
        # TODO also try with a one-hot encoding
        self.embed_layer = Embedding(
            voc_size,
            embed_dim,
            mask_zero=True,
            embeddings_initializer=lambda shape, dtype=None:
                tf.convert_to_tensor(
                    np.array(embed_init_w).reshape(shape),
                    dtype=dtype
                )
        )

        self.resizer = Resizing1D(
            length=sequence_target_length,
            interpolation="bilinear"
        )

        self.conv1 = Conv1D(
            conv_filters_1,
            kernel_size=conv_window_length_1,
            input_shape=(None, embed_dim),
            activation='relu',
            padding='valid',
            data_format="channels_last",
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(
                bias_vector_init_value),
            kernel_regularizer=tf.keras.regularizers.l2(
                penalty_regularization_w_conv)
        )
        self.globalmaxpool = GlobalMaxPooling1D()

        # Rectifier Layer, with dropout
        self.dense1 = Dense(
            dense_units_1,
            activation='relu',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(
                bias_vector_init_value)
        )
        self.dropout = Dropout(dropout_probability)

        # Classification head with softmax
        self.classifier = Dense(
            n_classes,
            activation='softmax',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(
                bias_vector_init_value)
        )

        # dropout won't be built on predict:
        self.dropout.build((None, dense_units_1))

    @tf.function
    def __transform_contig(self, contig, training):
        # input shape: num_nodes * node_length
        # tf.debugging.assert_shapes([
        #     (x, ("num_nodes", "node_length"))
        # ])
        # embed each integer-represented nucleotide:
        x = self.embed_layer(contig)
        # tf.debugging.assert_shapes([
        #     (x, ("num_nodes", "node_length", self.embed_layer.output_dim))
        # ])
        # resize nodes:
        x = self.resizer(x)
        tf.debugging.assert_shapes([
            (x, ("num_nodes", sequence_target_length,
             self.embed_layer.output_dim))
        ])
        # convolve over node lengths:
        x = self.conv1(x)
        tf.debugging.assert_shapes([
            (x, ("num_nodes", "node_length_resized", self.conv1.filters))
        ])
        # take the max along each node:
        x = self.globalmaxpool(x)
        tf.debugging.assert_shapes([
            (x, ("num_nodes", self.conv1.filters))
        ])
        x = self.dense1(x)
        tf.debugging.assert_shapes([
            (x, ("num_nodes", self.dense1.units))
        ])
        x = self.dropout(x, training=training)
        x = self.classifier(x)
        tf.debugging.assert_shapes([
            (x, ("num_nodes", self.n_classes))
        ])
        # now vote by averaging the predictions from each node:
        # (NB: need to reshape since we took out the batch dimension)
        x = self.globalmaxpool(tf.expand_dims(x, axis=0))
        # 1 * 2
        tf.debugging.assert_shapes([
            (x, (1, self.n_classes))
        ])
        # squeezing to scrap the dummy batch dimension:
        return tf.squeeze(x)

    @ tf.autograph.experimental.do_not_convert
    def __transform_contig_train(self, contig):
        return self.__transform_contig(contig, tf.constant(True, dtype=tf.bool))

    @ tf.autograph.experimental.do_not_convert
    def __transform_contig_eval(self, contig):
        return self.__transform_contig(contig, tf.constant(False, dtype=tf.bool))

    @ tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        """Returns y_pred as a `inputs.shape[0] * n_classes` tensor.

        Input shape is assumed to be `batch_size * num_nodes *
        node_length`, ie the output of a `contigParser` with `ndims=2`.

        Called by `__call__`, `predict`, `fit` and so on.
        """
        # input shape: batch_size * num_nodes * node_length
        # apply the model on each contig independently:
        if training:
            fn = self.__transform_contig_train
        else:
            fn = self.__transform_contig_eval
        return tf.map_fn(
            fn,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=[2], dtype=tf.float32)
        )

    def get_config(self):
        return {"n_classes": self.n_classes, "voc_size": self.voc_size, "name": self.name}
