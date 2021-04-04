from params import *

from preprocess_data import create_cpc_ds
from cpc_model import Predict_z, CPC, InfoNCE
from custom_training import train_cpc
from utils import configure_gpu_options

if set_memory_growth_tf:
    configure_gpu_options()

# Generate a dataset
print(f'[INFO] - Loading the dataset {path_data_cpc.split("/")[-2]}')
train_ds_cpc = create_cpc_ds()

# Model
cpc = CPC(
    data_generator_arguments["T"],
    data_generator_arguments["k"],
    data_generator_arguments["N"],
    z_dim,
    c_dim,
    enc_model,
    ar_model,
    Predict_z,
    encoder_args,
    ar_args,
    mixed_precision,
)

# print summary
for i in train_ds_cpc.take(1):
    cpc(i, training=False)
print(f"[INFO] - Created the {cpc.name} model.\n")
cpc.summary()

# load trained model to continue training when training was stopped intermediately
if path_to_continue_training:
    cpc.load_weights(path_to_continue_training)

# Loss and corresponding metric
loss = InfoNCE()
train_loss_metric_cpc = tf.keras.metrics.Mean("train_loss_CPC")

# Training
train_cpc(
    cpc,
    train_ds_cpc,
    loss,
    optimizer_cpc,
    epochs_cpc,
    steps_per_epoch_cpc,
    mixed_precision,
    path_save_cpc,
)
