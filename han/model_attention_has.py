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
            self.build_RNN()
            self.build_attention()

            # self.model_output_raw = self.rnn_output_raw
            self.model_output_raw = self.attention_output

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
            # print(self.input_layer.shape.as_list())
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
            # self.model_output = tf.reshape(self.model_output_flat, [self.batch_size_op, self.config['num_hidden_units']])

    def build_RNN(self):

        """
        Creates LSTM cell(s) and recurrent model.
        """

        # for bidirectional RNN uncomment that and comment second with ...
        # with tf.variable_scope("recurrent", reuse=self.reuse, initializer=self.initializer, regularizer=None):
        #     self.rnn_output, _ = bi_rnn(GRUCell(self.hidden_size), GRUCell(self.hidden_size),
        #                                 inputs=self.cnn_output, dtype=tf.float32)

        with tf.variable_scope(
            "recurrent",
            reuse=self.reuse,
            initializer=self.initializer,
            regularizer=None,
        ):

            rnn_cells = []
            for i in range(self.config["num_layers"]):
                rnn_cells.append(
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=self.config["num_hidden_units_RNN"]
                    )
                )

            if self.config["num_layers"] > 1:
                # Stack multiple cells.
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                    cells=rnn_cells, state_is_tuple=True
                )
            else:
                self.rnn_cell = rnn_cells[0]

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(
                cell=self.rnn_cell,
                inputs=self.cnn_output,
                dtype=tf.float32,
                sequence_length=tf.reshape(self.input_seq_len, [self.batch_size_op]),
                time_major=False,
                swap_memory=True,
            )

    def build_attention(self):

        with tf.name_scope("Attention_layer"):
            self.attention_output, self.alphas = self.attention(
                self.rnn_output, self.attention_size, return_alphas=True
            )

    def attention(self, inputs, att_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        """
        # Inpired from the article "Hierarchical Attention Networks for Document Classification"
        # https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

        # inputs - h_{it} for words in (5) or h_{i} - for sentences in (8)
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)
            print("inputs_shape", inputs.shape.as_list())

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        # D value - hidden size of the RNN layer
        hiddensize = inputs.shape[2].value

        # Trainable parameters
        # omega = word!!!
        # w_omega - W_{w} in (5)
        # b_omega - b_{w} in (5)
        # u_omega - context vector u_{w} in (6) for words or u_{s} in (9) for sentences,
        # u_omega on the main picture

        # Be cautios!!!! We need to be sure that these params are properly updated
        # initializer = tf.random_normal_initializer(stddev=0.1)
        w_omega = tf.get_variable(
            name="w_omega",
            shape=[hiddensize, att_size],
            initializer=self.initializer_attention,
        )
        b_omega = tf.get_variable(
            name="b_omega", shape=[att_size], initializer=self.initializer_attention
        )
        u_omega = tf.get_variable(
            name="u_omega", shape=[att_size], initializer=self.initializer_attention
        )

        with tf.name_scope("v"):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            # (5) expression
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        # in (6)
        vu = tf.tensordot(v, u_omega, axes=1, name="vu")  # (B,T) shape
        alphas = tf.nn.softmax(vu, name="alphas")  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        # (7) or (10) equation
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

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
