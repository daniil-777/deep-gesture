"""
Configuration file for local running of 3D-CNN
"""


import time
import os

config = dict()

"""
##################################################################
# Please note that the following fields will be set by our scripts to re-train and re-evaluate your model.

# Where experiment results are stored.
config['log_dir'] = './runs/'
# In case your pre/post-processing scripts generate intermediate results, you may use config['tmp_dir'] to store them.
config['tmp_dir'] = './tmp/'
# Path to training, validation and test data folders.
config['data_dir'] = '/cluster/project/infk/hilliges/lectures/mp19/project1'
##################################################################
# You can modify the rest or add new fields as you need.
"""

# Data
config["json_dir"] = "../data/sports1m_json/sports1m_test.json"
config["num_videos"] = 8
config["data_directory"] = "data/tf_records"
config["frame_height"] = 100
config["frame_width"] = 100
config["clip_size"] = 16
config["dropout_rate"] = 0.8
# Logs dir
config["logs_dir"] = "data/logs"

# Dataset statistics. You don't need to change unless you use different splits.
# config['num_test_samples'] =  0
# config['num_validation_samples'] = 0
# config['num_training_samples'] = 0

# Hyper-parameters and training configuration.
config["batch_size"] = 4
config["learning_rate"] = 1e-4
# Learning rate is annealed exponentially in 'exponential' case. You can change annealing configuration in the code.
config["learning_rate_type"] = "fixed"  # 'fixed' or 'exponential'

# config['num_steps_per_epoch'] =  int(config['num_training_samples']/config['batch_size'])

config["num_epochs"] = 8
config["evaluate_every_step"] = 32  # config['num_steps_per_epoch']
config["checkpoint_every_step"] = 10000000  # config['num_steps_per_epoch']*2
config["print_every_step"] = 16

# Dataset and Input Pipeline Configuration
config["inputs"] = {}
config["inputs"]["num_epochs"] = config["num_epochs"]
config["inputs"]["batch_size"] = config["batch_size"]

# 3D-CNN model parameters
config["3DCNN"] = {}
config["3DCNN"]["num_filters"] = [
    16,
    32,
    64,
]  # Number of filters for every convolutional layer.
config["3DCNN"]["filter_size"] = [3, 3, 3]  # Kernel size. Assuming kxk kernels.
config["3DCNN"][
    "num_hidden_units"
] = 32  # Number of units in the last dense layer, i.e. representation size.
config["3DCNN"]["num_class_labels"] = 2
config["3DCNN"]["batch_size"] = config["batch_size"]

# (2+1)D-CNN model parameters
config["2plus1DCNN"] = {}
config["2plus1DCNN"]["num_filters"] = [
    (16, 16),
    (32, 32),
    (64, 64),
]  # Number of filters for every convolutional layer.
config["2plus1DCNN"]["spatial_filter_size"] = [3, 3, 3]  # Spatial Kernel size.
config["2plus1DCNN"]["temporal_filter_size"] = [3, 3, 3]  # Temporal Kernel size.
config["2plus1DCNN"][
    "num_hidden_units"
] = 64  # Number of units in the last dense layer, i.e. representation size.
config["2plus1DCNN"]["num_class_labels"] = 2
config["2plus1DCNN"]["batch_size"] = config["batch_size"]

# RNN_attention model parameters
config["RNNModel_attention"] = {}
config["RNNModel_attention"]["attention_size"] = 50
config["RNNModel_attention"]["hidden_size"] = 150
config["RNNModel_attention"]["KEEP_PROB"] = 0.8
config["RNNModel_attention"]["num_class_labels"] = 2
config["RNNModel_attention"]["num_hidden_units"] = 64
config["RNNModel_attention"]["dropout_rate"] = 0.8
# You can set descriptive experiment names or simply set empty string ''.
config["model_name"] = "experiment-name"

# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
model_folder_name = (
    timestamp if config["model_name"] == "" else timestamp + "_" + config["model_name"]
)
config["model_id"] = timestamp
config["model_dir"] = os.path.abspath(
    os.path.join(config["logs_dir"], model_folder_name)
)
print("Writing to {}\n".format(config["model_dir"]))
