# handling logging level
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging

logging.getLogger("tensorflow").setLevel(logging.WARNING)

# general imports
import tensorflow as tf
from datetime import datetime
from utils import set_mixed_precission, get_command_line_args, check_dirs

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
    project_path = os.getcwd()  # path to the project folder
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

# create an appropriate directory structure
check_dirs(
    [
        path_save_embeddings_train,
        path_save_embeddings_test,
        path_save_classifier_plots,
        path_save_cpc,
    ]
)


# ------------- CPC training params ---------------------------------

epochs_cpc = 700
steps_per_epoch_cpc = 100  # how many batch are fed in a single epoch
optimizer_cpc = tf.keras.optimizers.Adam(1e-5)


# --------------- Classifier Training params-------------

epochs_class = 10
optimizer_class = tf.keras.optimizers.Adam(1e-4)
batch_size_class = 32
test_size_classifier = 5000  # number of samples to be used as a test dataset


# --------------- Dimension Reduction params (for classifier)----

r_dim = 32  # num dimensions to reduce to
epochs_dimension_reduction = 10
optimizer_dimension_reduction = tf.keras.optimizers.Adam(1e-5)
batch_size_dimension_reduction = 32


# -------------- generate embeddings params --------------------------

# How often to randomly sample from a single data to get different audio segments of length
num_em_samples_per_train_data = 90
num_em_samples_per_test_data = 5


# --------------- TSNE embedding visualization parameters -------------------------------

num_tsne = 3000  # number of points to plot in t-SNE


# -------------- encoder params -----------------------------------

z_dim = 256  # dimension of latent representation

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
        "T": 27,  # num. timestep until current
        "k": 3,  # num. timestep to predict
        "N": 8,  # num. samples (N-1 = num. negative samples)
        "full_duration": 4,  # sec, total length of a single sequence
        "original_sr": 22050,  # Hz, sampling rate of original data
        "desired_sr": 4410,  # Hz, sampling rate that is desired, used to down sample
        "data_path": path_data_cpc,  # str, where to get raw audio data from
    }

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
        "stride_sizes": [2, 2, 2, 2],
        "kernel_sizes": [3, 3, 3, 3],
        "n_filters": [32, 64, 256, 512],
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
        "data_path": path_data_cpc,  # str
    }


# ----------------- Parameters for Autoregressive Model --------------------------------

c_dim = 512 # size of context embedding vector

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
        + data_generator_arguments["k"],  # maximum length of a single sequence in steps
        "rate": 0.1,  # dropout rate
    }
