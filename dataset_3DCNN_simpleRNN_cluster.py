"""
Dataset class original data for 3DCNN + Simple RNN

Class for reading input data, stored in TFRecords.
"""


import tensorflow as tf
import numpy as np
import functools


class Dataset_3DCNN_plus_Simple_RNN:
    """
    Dataset class for reading TFRecord files. You can implement
    """

    def __init__(
        self,
        data_path,
        batch_size,
        window_size=3,
        normalize=False,
        shuffle=True,
        num_parallel_calls=4,
        **kwargs
    ):
        # To reshape the serialized data. Do not change these values unless you create tfrecords yourself and have
        # different size.
        self.RGB_SIZE = (-1, 80, 80, 3)
        self.DEPTH_SIZE = (-1, 80, 80, 1)
        self.BINARY_SIZE = (-1, 80, 80, 3)
        self.SKELETON_SIZE = (-1, 180)

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parallel_calls = num_parallel_calls
        self.normalize = normalize
        self.tf_data = None

        # self.dataset_TFRecord = dataset_TFRecord
        # self.frame_height = frame_height
        # self.frame_width = frame_width
        # self.num_channels = num_channels
        self.clip_size = None  # set dynamically
        # self.batch_size = batch_size
        # self.shuffle = shuffle

        self.window_size = window_size

        # self.consume_data()

        self.tf_data_transformations()
        self.tf_data_to_model()

        if tf.executing_eagerly():
            self.iterator = self.tf_data.make_one_shot_iterator()
            self.tf_samples = None
        else:
            self.iterator = self.tf_data.make_initializable_iterator()
            self.tf_samples = self.iterator.get_next()

    def get_iterator(self):
        return self.iterator

    def get_tf_samples(self):
        return self.tf_samples

    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        tf_data_opt = tf.data.Options()
        # tf_data_opt.experimental_autotune = True

        tf_data_files = tf.data.Dataset.list_files(
            self.data_path, seed=1234, shuffle=self.shuffle
        )
        self.tf_data = tf.data.TFRecordDataset(
            filenames=tf_data_files,
            compression_type="ZLIB",
            num_parallel_reads=self.num_parallel_calls,
        )
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.map(
            functools.partial(self.__parse_single_tfexample_fn),
            num_parallel_calls=self.num_parallel_calls,
        )
        self.tf_data = self.tf_data.prefetch(self.batch_size * 10)

        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(self.batch_size * 10)
        if self.normalize:
            self.tf_data = self.tf_data.map(
                functools.partial(self.__normalize_with_local_stats),
                num_parallel_calls=self.num_parallel_calls,
            )

    def tf_data_to_model(self):
        self.tf_data = self.tf_data.padded_batch(
            self.batch_size, padded_shapes=self.tf_data.output_shapes
        )
        self.tf_data = self.tf_data.prefetch(2)

    def __normalize_with_local_stats(self, tf_sample_dict):
        """
        Given a sample dictionary (see __parse_single_tfexample_fn return), calculates mean and std and applies
        zero-mean, unit-variance standardization.
        """

        def get_mean_and_std(tensor, keepdims=False):
            """
            Calculates mean and standard deviation of a tensor over given dimensions.
            """
            mean = tf.reduce_mean(tensor, keepdims=True)
            diff_squared = tf.square(tensor - mean)
            variance = tf.reduce_mean(diff_squared, keepdims=keepdims)
            std = tf.maximum(tf.sqrt(variance), 1e-6)
            return mean, std

        rgb_mean, rgb_std = get_mean_and_std(tf_sample_dict["rgb"])
        tf_sample_dict["rgb"] = (tf_sample_dict["rgb"] - rgb_mean) / rgb_std

        return tf_sample_dict

    def __parse_single_tfexample_fn(self, proto):
        feature_to_type = {
            "rgb": tf.FixedLenFeature([], dtype=tf.string),
            "depth": tf.FixedLenFeature([], dtype=tf.string),
            "segmentation": tf.FixedLenFeature([], dtype=tf.string),
            "skeleton": tf.FixedLenFeature([], dtype=tf.string),
            "length": tf.FixedLenFeature([1], dtype=tf.int64),
            "label": tf.FixedLenFeature([1], dtype=tf.int64),
            "id": tf.FixedLenFeature([1], dtype=tf.int64),
        }

        features = tf.parse_single_example(proto, feature_to_type)
        features["rgb"] = tf.reshape(
            tf.decode_raw(features["rgb"], tf.float32), self.RGB_SIZE
        )

        # clip size is set dynamically
        self.clip_size = tf.shape(features["rgb"])[0]

        stacked_sample = tf.zeros(
            [
                self.clip_size - self.clip_size,
                self.window_size,
                self.RGB_SIZE[1],
                self.RGB_SIZE[2],
                self.RGB_SIZE[3],
            ]
        )

        # -----This part is needed to have a window of frames for each time step t-----

        for t in range(self.clip_size - self.window_size + 1):
            one_sample = tf.slice(
                features["rgb"],
                [t, 0, 0, 0],
                [
                    self.window_size,
                    self.RGB_SIZE[1],
                    self.RGB_SIZE[2],
                    self.RGB_SIZE[3],
                ],
            )
            if t == 0:
                stacked_sample = tf.expand_dims(one_sample, 0)
            else:
                stacked_sample = tf.concat(
                    [stacked_sample, tf.expand_dims(one_sample, axis=0)], axis=0
                )
        # -------------------------------------------------------------------------------------

        """  
        # SUBSTITUTE TO THE CORRESPONDING SNIPPET JUST ABOVE TO BE ABLE TO SKIP SOME AMOUNT 
        # OF FRAMES ("skip" parameter) BETWEEN TWO FRAMES  
        # -----This part is needed to have a window of frames for each time step t-----
        # change step to control how many frames to skip
        skip = 8
        t = 0
        while (t < (self.clip_size - self.window_size + 1)):
            one_sample = tf.slice(features["rgb"], [t, 0, 0, 0],
                                  [self.window_size, self.RGB_SIZE[1], self.RGB_SIZE[2], self.RGB_SIZE[3]])
            if t == 0:
                stacked_sample = tf.expand_dims(one_sample, 0)
            else:
                stacked_sample = tf.concat([stacked_sample, tf.expand_dims(one_sample, axis=0)], axis=0)
            t += skip
        # -------------------------------------------------------------------------------------
        """
        # These are the only necessary ones --------------------------------------------------
        features["features"] = stacked_sample
        length = tf.cast(tf.shape(stacked_sample)[0], dtype=tf.int32)
        length = tf.reshape(length, [1])
        features["length"] = length[0]
        features["label"] = features["label"][0]
        # -------------------------------------------------------------------------------------

        features["depth"] = tf.reshape(
            tf.decode_raw(features["depth"], tf.float32), self.DEPTH_SIZE
        )
        features["segmentation"] = tf.reshape(
            tf.decode_raw(features["segmentation"], tf.float32), self.BINARY_SIZE
        )
        features["skeleton"] = tf.reshape(
            tf.decode_raw(features["skeleton"], tf.float32), self.SKELETON_SIZE
        )
        features["id"] = features["id"][0]

        return features
