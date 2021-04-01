import numpy as np
import tensorflow as tf
from datetime import datetime


@tf.function
def train_step(model, ds, loss_function, optimizer,
               steps_per_epoch, train_loss_metric=None, mixed_precision=False):
    for batch in ds.take(steps_per_epoch):

        with tf.GradientTape() as tape:

            prediction = model(batch, training=True)
            loss = loss_function(prediction) + tf.reduce_sum(model.losses)  # l2 reg

            if mixed_precision:
                loss = optimizer.get_scaled_loss(loss)  # scaled loss for mixed precision training

        gradients = tape.gradient(loss, model.trainable_variables)  # get (scaled) gradients

        if mixed_precision:
            gradients = optimizer.get_unscaled_gradients(gradients)  # get unscaled gradients from scaled gradients

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # apply unscaled gradients

        # update metric
        train_loss_metric.update_state(loss)


# formerly main eval train with mode (now we only train CPC with a custom train function and the classifier with model.fit()
def train_cpc(cpc_model, train_ds, loss_function, optimizer, epochs, steps_per_epoch, mixed_precision, save_to):
    train_losses = []
    train_loss_metric = tf.keras.metrics.Mean()
    for e in range(epochs):
        train_step(cpc_model, train_ds, loss_function, optimizer, steps_per_epoch, train_loss_metric, mixed_precision)

        train_losses.append(train_loss_metric.result().numpy())

        print(f"Episode:{e}    loss: {train_losses[-1]}")

        train_loss_metric.reset_states()

    now = datetime.now()

    # save model parameters to .h5 file. Can afterwards be loaded with cpc.load_weights(load_from)
    save_to = save_to + str(now)[:-10] + ".h5"
    cpc_model.save_weights(save_to, overwrite=False)

    # save loss array for later visualization
    losses_array = np.array(train_losses)
    np.save(save_to.replace(".h5", ".npy"), losses_array)
