# handling comand line arguments
import argparse

# handling logging level
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

# general imports
from datetime import datetime
from utils import set_mixed_precission

# use mixed precision allows for bigger models
mixed_precision = False
set_mixed_precission(mixed_precision)

# ----------- setting Path variables-----------------------------
cmd_args = get_command_line_args()
modelname = cmd_args[
    "model_name"
]  # one of '1dconv_gru/' '1dconv_gru/', '2dconv_gru/', '2dconv_transformer/'
mode = "local"  # one of 'colab', 'local'
if mode == "local":
    project_path = os.path.dirname(os.getcwd())  # path to the project folder
    set_memory_growth_tf = True
else:  # colab
    project_path = "/content/final/"
    set_memory_growth_tf = False

# location to load intermediate weights if training wasn't done until the end
path_to_continue_training = False

# assigning remaining path variables
# location to save weights and loss results from cpc training
path_save_cpc = os.path.join(project_path, "results/cpc/", modelname)
# location to load saved cpc model weights
path_load_model = os.path.join(project_path, "results/cpc/", modelname, "weights.h5")
# location to save generated embeddings using trained cpc
path_save_embeddings_train = os.path.join(
    project_path, "results/embeddings/", modelname, "train/"
)
path_save_embeddings_test = os.path.join(
    project_path, "results/embeddings/", modelname, "test/"
)
# location to load saved embeddings
path_load_embeddings_train = os.path.join(
    project_path, "results/embeddings/", modelname, "train/"
)
path_load_embeddings_test = os.path.join(
    project_path, "results/embeddings/", modelname, "test/"
)
# location to save figures and history of classifier
current_time = datetime.now()
datetime_str = current_time.strftime("%Y%m%d-%H%M%S")
path_save_classifier_plots = os.path.join(
    project_path, "results/classifier/", modelname, datetime_str
)

check_dirs(
    [
        path_save_embeddings_train,
        path_save_embeddings_test,
        path_save_classifier_plots,
        path_save_cpc,
    ]
)

# ------------- training params ---------------------------------
epochs_cpc = 700
steps_per_epoch_cpc = 100
epochs_class = 1000
learning_rate = 1e-5  # for cpc
learning_rate_class = 1e-3  # for classifier
batch_size_classifier = 32

# -------------- classifier data params --------------------------
# How often to sample from a single data to get different parts
num_em_samples_per_train_data = 10
num_em_samples_per_test_data = 1

# -------------- encoder params -----------------------------------
z_dim = 256

# 1dconv encoder params (raw audio data)
if modelname == "1dconv_transformer/" or modelname == "1dconv_gru/":
    enc_model = "1d_conv"
    # location of raw audio to train cpc
    path_data_cpc = os.path.join(project_path, "data/fma_small_mono_wav/")
    # location of raw audio to generate embeddings
    path_data_train = os.path.join(project_path, "data/GTZAN/")
    path_data_test = os.path.join(project_path, "data/test_data/")
    encoder_args = {
        "z_dim": z_dim,
        "stride_sizes": [5, 4, 2, 2, 2],
        "kernel_sizes": [10, 8, 4, 4, 4],
        "n_filters": [512, 512, 512, 512, 512],
        "activation": tf.nn.leaky_relu,
    }
    data_generator_arguments = {
        "T": 27,  # timestep
        "k": 3,  # timestep
        "N": 8,  # number
        "full_duration": 4,  # sec
        "original_sr": 22050,  # Hz
        "desired_sr": 4410,  # Hz
        "data_path": path_data_cpc,
    }  # str

# 2dconv encoder params (mel spectogram data)
elif modelname == "2dconv_gru/" or modelname == "2dconv_transformer/":
    enc_model = "2d_conv"
    # location of raw audio to train cpc
    path_data_cpc = os.path.join(project_path, "data/fma_mel_specs/")
    # location of raw audio to generate embeddings
    path_data_train = os.path.join(project_path, "data/GTZAN_mel_specs/")
    path_data_test = os.path.join(project_path, "data/test_data_mel_specs/")
    encoder_args = {
        "z_dim": z_dim,
        "stride_sizes": [1, 2, 1, 2, 2, 2],
        "kernel_sizes": [5, 3, 5, 3, 3, 3],
        "n_filters": [128, 128, 256, 256, 512, 512],
        "dense_units": [512],
        "conv_fct": tf.nn.leaky_relu,
        "dense_act": tf.nn.leaky_relu,
        "kernel_reg": True,
    }

    data_generator_arguments = {
        "T": 27,  # timestep
        "k": 3,  # timestep
        "N": 8,  # timestep
        "batch_size": 8,
        "data_path": path_data_cpc,
    }  # str


# ----------------- AR params --------------------------------
c_dim = 512

# GRU AR params
if modelname == "1dconv_gru/" or modelname == "2dconv_gru/":
    ar_model = "GRU"
    ar_args = {}
# attentionAR params
elif modelname == "1dconv_transformer/" or modelname == "2dconv_transformer/":
    ar_model = "transformer"
    ar_args = {
        "num_enc_layers": 2,  # num. transformer encoder blocks
        "num_heads": 2,  # num. multiheads for attention
        "z_dim": z_dim,
        "dff": 100,  # num. units for 1st ffn within encoder block
        "dense_units": [100, c_dim],  # num. units for additional ffn
        "activation": tf.nn.leaky_relu,  # activation for additional ffn
        "maximum_position_encoding": data_generator_arguments["T"]
        + data_generator_arguments["k"],
        "rate": 0.1,
    }  # dropout rate
