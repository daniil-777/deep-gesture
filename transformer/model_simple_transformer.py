"""
Model 3DCNN + Simple RNN
"""


# TODO: Max Pooling layer for (2+1)D CNN:
# we need to understand better and to wrap our heads about the details.
# like why that stride for example.


import tensorflow as tf


class Complete_Model:
    """
    3DCNN plus simple RNN
    """

    def __init__(self, config, placeholders, mode):
        """

        :param config: dictionary of model parameters. configuration file
        :param placeholders: dictionary of input placeholders
        :param mode: running mode (training, validation, testing)
        """
        self.config = config
        self.input_placeholders = placeholders

        # running modality
        assert mode in ["training", "validation", "test"]
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"

        self.input_seq_len = placeholders[
            "length"
        ]  # Note this is a vector of size batch_size

        if self.mode is not "test":
            # self.input_target_labels = placeholders['label']
            self.input_target_labels = tf.cast(placeholders["label"], tf.int64)

        # Total number of trainable parameters.
        self.num_parameters = 0

        # Batch size and sequence length that are set dynamically.
        self.batch_size_op = None
        # self.seq_len_op = None
        self.input_rgb = placeholders["features"]
        # Training objective.
        self.loss = None
        # Logits.
        self.logits = None
        # Label prediction, i.e., what we need to make the submission.
        self.predictions = None
        # Number of the correct predictions.
        self.num_correct_predictions = None
        # Accuracy of the batch.
        self.batch_accuracy = None

        # Set by build_graph method.
        self.input_layer = None

        # CONVOLUTION
        # 3DCNN output (before dense layer)
        self.cnn_output_raw = None
        # 3DCNN flattened output  [batch_size*seq_len, CNN_representation_size]
        self.cnn_output_flat = None
        # 3DCNN output  [batch_size, seq_len, CNN_representation_size]
        self.cnn_output = None

        # Recurrancy
        self.rnn_output = None

        # Model outputs with shape [batch_size, seq_len, representation_size]
        self.model_output_raw = None
        # Model outputs with shape [batch_size*seq_len, representation_size]
        self.model_output_flat = None
        self.model_output = None
        # ATTENTION
        self.hidden_size = self.config["hidden_size"]
        self.attention_size = self.config["attention_size"]
        self.attention_output = None
        self.alphas = None
        self.initializer_attention = tf.random_normal_initializer(stddev=0.1)

        # Transformer parameters
        self.embed_dim = self.config["embed_dim"]
        self.num_heads = self.config["num_heads"]
        self.ff_dim = self.config["ff_dim"]
        self.transformer_hidden_output = self.config[
            "transformer_hidden_output"
        ]  # output before dense layer to the number of classes
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {self.embed_dim} should be divisible by number of heads = {self.num_heads}"
            )
        self.projection_dim = self.embed_dim // self.num_heads
        self.query_dense = tf.layers.Dense(self.embed_dim)
        self.key_dense = tf.layers.Dense(self.embed_dim)
        self.value_dense = tf.layers.Dense(self.embed_dim)
        self.combine_heads = tf.layers.Dense(self.embed_dim)

        # self.layernorm1 = tf.contrib.layers.layer_norm()
        # self.layernorm2 = tf.contrib.layers.layer_norm()
        self.rate = 0.1
        self.dropout1 = tf.layers.Dropout(self.rate)
        self.dropout2 = tf.layers.Dropout(self.rate)

        self.transformer_output = None

        self.initializer = tf.glorot_normal_initializer()

    def build_graph(self):
        with tf.variable_scope(
            "graph", reuse=self.reuse, initializer=self.initializer, regularizer=None
        ):
            self.input_layer = self.input_rgb
            # Inputs of the network have shape [batch_size, seq_len, window_size, height, width, num_channels]
            # We reshape to [batch_size*seq_len, window_size, height, width, num_channels]

            _, _, window_size, height, width, num_channels = self.input_layer.shape
            # tf.shape(<op>) provides dimensions dynamically that we know only at run-time.
            self.batch_size_op = tf.shape(self.input_layer)[0]
            self.seq_len_op = tf.shape(self.input_layer)[1]

            self.input_layer = tf.reshape(
                self.input_layer, [-1, window_size, height, width, num_channels]
            )
            self.build_3DCNN()

            self.build_transformer()

            # self.model_output_raw = self.attention_output
            self.model_output_raw = self.transformer_output
            # self.build_RNN()
            # self.build_attention()

            # # self.model_output_raw = self.rnn_output_raw
            # self.model_output_raw = self.attention_output

            # Shape of [batch_size, seq_len, representation_size]
            _, representation_size = self.model_output_raw.shape.as_list()

            self.model_output = self.model_output_raw
            self.model_output_flat = tf.reshape(
                self.model_output_raw, [-1, representation_size]
            )

    def build_3DCNN(self):
        """
        Stacks convolutional layers where each layer consists of (2+1)D convolution operations.
        re-lu activation (both temporal and spatial)
        """
        with tf.variable_scope(
            "2plus1Dconvolution",
            reuse=self.reuse,
            initializer=self.initializer,
            regularizer=None,
        ):
            input_layer_ = self.input_layer
            for i, num_filter in enumerate(self.config["num_filters"]):
                spatial_num_filter = num_filter[0]
                temporal_num_filter = num_filter[1]

                # spatial convolution
                spatial_conv_layer = tf.layers.conv3d(
                    inputs=input_layer_,
                    filters=spatial_num_filter,
                    kernel_size=[
                        1,
                        self.config["spatial_filter_size"][i],
                        self.config["spatial_filter_size"][i],
                    ],
                    padding="same",
                    activation=tf.nn.relu,
                )
                # temporal convolution
                temporal_conv_layer = tf.layers.conv3d(
                    inputs=spatial_conv_layer,
                    filters=temporal_num_filter,
                    kernel_size=[self.config["temporal_filter_size"][i], 1, 1],
                    padding="same",
                    activation=tf.nn.relu,
                )

                pooling_layer = tf.layers.max_pooling3d(
                    inputs=temporal_conv_layer,
                    pool_size=[2, 2, 2],
                    strides=[2, 2, 2],
                    padding="same",
                )
                input_layer_ = pooling_layer

            self.cnn_output_raw = input_layer_
            (
                batch_size_by_seq_length,
                new_temporal_dim,
                cnn_height,
                cnn_width,
                num_filters,
            ) = self.cnn_output_raw.shape.as_list()
            self.cnn_output_flat = tf.reshape(
                self.cnn_output_raw,
                [-1, new_temporal_dim * cnn_height * cnn_width * num_filters],
            )

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            self.cnn_output_flat = tf.layers.dense(
                inputs=self.cnn_output_flat,
                units=self.config["num_hidden_units"],
                activation=tf.nn.relu,
            )
            # CNN OUTPUT FLAT shape [batch_size * seq_len, num_hidden_units]
            # CNN OUTPUT shape [batch_size, seq_len, num_hidden_units]
            self.cnn_output = tf.reshape(
                self.cnn_output_flat,
                [self.batch_size_op, self.seq_len_op, self.config["num_hidden_units"]],
            )

    def build_loss(self):
        """
        Calculates classification loss depending on loss type. We are trying to assign a class label to input
        sequences (i.e., many to one mapping). We need to reduce sequence information into a single step by either
        selecting the last step or taking average over all steps. You are welcome to implement a more sophisticated
        approach.
        """
        self.seq_loss_mask = tf.expand_dims(
            tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1
        )
        # Calculate logits
        with tf.variable_scope(
            "logits", reuse=self.reuse, initializer=self.initializer, regularizer=None
        ):
            dropout_layer = tf.layers.dropout(
                inputs=self.model_output_flat,
                rate=self.config["dropout_rate"],
                training=self.is_training,
            )
            logits_non_temporal = tf.layers.dense(
                inputs=dropout_layer, units=self.config["num_class_labels"]
            )
            self.logits = tf.reshape(
                logits_non_temporal,
                [self.batch_size_op, -1, self.config["num_class_labels"]],
            )
            print("logits shape", self.logits.shape.as_list())

        with tf.variable_scope(
            "logits_prediction",
            reuse=self.reuse,
            initializer=self.initializer,
            regularizer=None,
        ):
            # Select the last step. Note that we have variable-length and padded sequences.
            if self.config["loss_type"] == "last_logit":
                self.logits = tf.gather_nd(
                    self.logits,
                    tf.stack(
                        [tf.range(self.batch_size_op), self.input_seq_len - 1], axis=1
                    ),
                )
                accuracy_logit = self.logits
            # Take average of time steps.
            elif self.config["loss_type"] == "average_logit":
                self.logits = tf.reduce_mean(self.logits * self.seq_loss_mask, axis=1)
                accuracy_logit = self.logits
            elif self.config["loss_type"] == "average_loss":
                accuracy_logit = tf.reduce_mean(
                    self.logits * self.seq_loss_mask, axis=1
                )
            else:
                raise Exception("Invalid loss type")

        if self.mode is not "test":
            with tf.name_scope("cross_entropy_loss"):
                if self.config["loss_type"] == "average_loss":
                    labels_all_steps = tf.tile(
                        tf.expand_dims(self.input_target_labels, dim=1),
                        [1, tf.reduce_max(self.input_seq_len)],
                    )
                    self.loss = tf.contrib.seq2seq.sequence_loss(
                        logits=self.logits,
                        targets=labels_all_steps,
                        weights=self.seq_loss_mask[:, :, 0],
                        average_across_timesteps=True,
                        average_across_batch=True,
                    )
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits, labels=self.input_target_labels
                        )
                    )

            with tf.name_scope("accuracy_stats"):
                # Return a bool tensor with shape [batch_size] that is true for the correct predictions.
                correct_predictions = tf.equal(
                    tf.argmax(accuracy_logit, 1), self.input_target_labels
                )
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(
                    tf.cast(correct_predictions, tf.int64)
                )
                # Calculate the accuracy per mini-batch.
                self.batch_accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32)
                )

        # Accuracy calculation.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            self.predictions = tf.argmax(accuracy_logit, 1, name="predictions")

    def get_num_parameters(self):
        """
        :return: total number of trainable parameters.
        """
        # Iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # multiplying dimension values
            self.num_parameters += local_parameters

        return self.num_parameters

    ##########################transformer############################################
    # Multi-Head mechanism is from "Attention Is All You Need"
    # https://arxiv.org/pdf/1706.03762.pdf

    def attention_transformer(self, query, key, value):
        """calculates attention using query, key, value"""
        # z = softmax(Q*K^{T}/sqrt(d_{K}) * V)
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """distributes results of attentions over heads"""
        # Q0 K0 V0, Q1 K1 V1...
        #    |        |             |
        #    z0       z1   ... z_{num_heads}
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def transformer_block(self, inputs):
        """
        Encoder from transformer
        inputs - tensor [None, seq_len (from cnn), in_dim_from_cnn]
        """

        embed_dim = self.embed_dim  # Embedding size for each token
        num_heads = self.num_heads  # Number of attention heads
        ff_dim = (
            self.ff_dim
        )  # Hidden layer size in feed forward network inside transformer

        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        # batch_size = self.batch_size_op
        #########################begin of attention transformer part################################

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)

        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention_transformer(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)

        # splitting heads together through a dense layer
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)

        attn_output = self.dropout1(output, training=self.is_training)
        # normalisation in the residual of input to attention block and Multi-Head attention
        attn_output_norm = tf.contrib.layers.layer_norm(inputs + attn_output)
        #########################end of attention transformer part################################
        ffn_output = tf.layers.dense(
            inputs=attn_output_norm, units=ff_dim, activation="relu"
        )
        ffn_output = tf.layers.dense(
            inputs=ffn_output, units=embed_dim, activation="relu"
        )

        ffn_output = self.dropout2(ffn_output, training=self.is_training)
        # OUTPUT (batch_size, seq_len, embed_dim)

        # normalisation in the end of attention block
        ffn_output = tf.contrib.layers.layer_norm(attn_output_norm + ffn_output)
        ffn_output = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        # ffn_output = tf.reshape(ffn_output, [batch, seq])

        # FC layers for output classes
        # you may remove droput to tune parameters
        ffn_output = tf.layers.Dropout(0.1)(ffn_output)
        ffn_output = tf.layers.Dense(self.transformer_hidden_output, activation="relu")(
            ffn_output
        )
        ffn_output = tf.layers.Dropout(0.1)(ffn_output)
        # shape [Batch, representation_size_transformer]
        return ffn_output

    def build_transformer(self):
        with tf.name_scope("Transformer_layer"):
            self.transformer_output = self.transformer_block(inputs=self.cnn_output)
