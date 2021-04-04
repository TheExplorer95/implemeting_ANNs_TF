from params import *
import numpy as np
import tensorflow as tf
import os
from utils import Timer


@tf.function
def train_step(
    model,
    ds,
    loss_function,
    optimizer,
    steps_per_epoch,
    train_loss_metric,
    mixed_precision=False,
):
    """
    perform a weight update for a single epoch.
    :param model: tf.keras.Model
    :param ds: tf.keras.data.Dataset
    :param loss_function: tf.keras.loss
    :param optimizer: tf.keras.optimizers
    :param steps_per_epoch: int, num. batch to pass per epoch
    :param train_loss_metric: tf.keras.metrics, metric to log train loss
    :param mixed_precision: boolean, allow mixed_precision for performance boost if True
    :return: None
    """
    for batch in ds.take(steps_per_epoch):

        # forward pass with GradientTape
        with tf.GradientTape() as tape:
            prediction = model(batch, training=True)
            loss = loss_function(prediction) + tf.reduce_sum(model.losses)  # l2 reg

            if mixed_precision:
                loss = optimizer.get_scaled_loss(
                    loss
                )  # scaled loss for mixed precision training

        # backward pass via GradienTape (auto-gradient calc)
        if not tf.math.is_nan(loss) and tf.math.is_finite(loss):
            gradients = tape.gradient(
                loss, model.trainable_variables
            )  # get (scaled) gradients
            if mixed_precision:
                gradients = optimizer.get_unscaled_gradients(
                    gradients
                )  # get unscaled gradients from scaled gradients
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )  # apply unscaled gradients

            # update metric
            train_loss_metric.update_state(loss)


def train_cpc(
    cpc_model,
    train_ds,
    loss_function,
    optimizer,
    epochs,
    steps_per_epoch,
    mixed_precision,
    save_path,
    print_interval=10,
    checkpoint_interval=100,
):
    """
    Train a CPC model and save weights
    :param cpc_model: tf.keras.model
    :param train_ds: tf.keras.data.dataset
    :param loss_function: tf.keras.loss
    :param optimizer: tf.keras.optimizers
    :param epochs: int, num. epochs to train for
    :param steps_per_epoch: int, num. batch to pass for a single epoch
    :param mixed_precision: boolean, allow mixed precision for performance boost if True
    :param save_path: str, path to save weights
    :param print_interval: int, in epochs, how often train progress is printed
    :param checkpoint_interval: int, in epochs, how often weights are saved intermediately
    :return: None
    """
    timer = Timer()
    times = []
    train_losses = []
    train_loss_metric = tf.keras.metrics.Mean()

    print(f"\n[Info] - Started training the model.")
    for e in range(epochs):
        timer.start()
        train_step(
            cpc_model,
            train_ds,
            loss_function,
            optimizer,
            steps_per_epoch,
            train_loss_metric,
            mixed_precision,
        )

        # evaluate metrics
        train_losses.append(train_loss_metric.result().numpy())
        train_loss_metric.reset_states()
        elapsed_time = timer.stop()
        times.append(elapsed_time)

        # fancy printing
        print(f"[Epoch {e}] - dT: {elapsed_time:0.2f}s - loss: {train_losses[-1]:0.6f}")
        if e % print_interval == 0:
            print(
                f"\n[INFO] - Total time elapsed: {np.sum(times)/60:0.2f} min. Total time remaining: {(np.sum(times)/(e+1))*(epochs-e-1)/60: 0.2f} min.\n"
            )
        elif e % checkpoint_interval == 0:
            # save the weights every checkpoint_interval
            model_fn = f"checkpoint_episode{e}_weights.h5"
            cpc_model.save_weights(os.path.join(save_path, model_fn), overwrite=True)

    # save model parameters to .h5 file. Can afterwards be loaded with cpc.load_weights(load_from)
    model_fn = "weights.h5"
    cpc_model.save_weights(os.path.join(save_path, model_fn), overwrite=False)

    # save loss array for later visualization
    losses_array = np.array(train_losses)
    loss_fn = "loss_data.npy"
    np.save(os.path.join(save_path, loss_fn), losses_array)
