import numpy as np
import tensorflow as tf
import os
import time


class Timer:
    # A small class for making timings.
    def __init__(self):
        self._start_time = None

    def start(self):
        # Start a new timer
        self._start_time = time.perf_counter()

    def stop(self):
        # Stop the timer, and report the elapsed time
        if self._start_time is None:
            print("Timer is not running. Use .start() to start it")
            return 0

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time


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
        if not tf.math.is_nan(loss):
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


# formerly main eval train with mode (now we only train CPC with a custom train function and the classifier with model.fit()
def train_cpc(
    cpc_model,
    train_ds,
    loss_function,
    optimizer,
    epochs,
    steps_per_epoch,
    mixed_precision,
    save_path,
):
    timer = Timer()
    times = []
    train_losses = []
    train_loss_metric = tf.keras.metrics.Mean()
    print(f"[Info] - Started training the model.")
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
        if e % 10 == 0 and not e == 0:
            print(
                f"\n[INFO] - Total time elapsed: {np.sum(times)/60:0.2f} min. Total time remaining: {(np.sum(times)/(e+1))*(epochs-e-1)/60: 0.2f} min.\n"
            )

        print(
            f"[Epoch {e}] - dT: {elapsed_time:0.2f}s - loss: {train_losses[-1]:0.6f}"
        )

    # save model parameters to .h5 file. Can afterwards be loaded with cpc.load_weights(load_from)
    model_fn = "weights.h5"
    cpc_model.save_weights(os.path.join(save_path, model_fn), overwrite=False)

    # save loss array for later visualization
    losses_array = np.array(train_losses)
    loss_fn = "loss_data.npy"
    np.save(os.path.join(save_path, loss_fn), losses_array)
