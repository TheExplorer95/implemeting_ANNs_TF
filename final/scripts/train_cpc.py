from params import *
from preprocess_data import create_cpc_ds
from cpc_model import Predict_z, CPC, InfoNCE
from training import train_cpc

# Generate a dataset
train_ds_cpc = create_cpc_ds()

# Define 3 design components
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
for i in train_ds_cpc.take(1):
    cpc(i, training=False)
cpc.summary()
# load trained model
if path_to_continue_training:
    cpc.load_weights(path_to_continue_training)

# Loss
loss = InfoNCE()
train_loss_metric_cpc = tf.keras.metrics.Mean("train_loss_CPC")
# Optimizer
adam = tf.keras.optimizers.Adam(learning_rate)

# Training
train_cpc(
    cpc,
    train_ds_cpc,
    loss,
    adam,
    epochs_cpc,
    steps_per_epoch_cpc,
    mixed_precision,
    path_save_cpc,
)
