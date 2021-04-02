import numpy as np
import tensorflow as tf
import os

@tf.function
def train_step(model, ds, loss_function, optimizer,
               steps_per_epoch, train_loss_metric, mixed_precision=False):
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
def train_cpc(cpc_model, train_ds, loss_function, optimizer, epochs, steps_per_epoch, mixed_precision, save_path):
    train_losses = []
    train_loss_metric = tf.keras.metrics.Mean()
    for e in range(epochs):
        train_step(cpc_model, train_ds, loss_function, optimizer, steps_per_epoch, train_loss_metric, mixed_precision)

        train_losses.append(train_loss_metric.result().numpy())

        print(f"Episode:{e}    loss: {train_losses[-1]}")

        train_loss_metric.reset_states()

    # save model parameters to .h5 file. Can afterwards be loaded with cpc.load_weights(load_from)
    model_fn = 'weights.h5'
    cpc_model.save_weights(os.path.join(save_path, model_fn), overwrite=False)

    # save loss array for later visualization
    losses_array = np.array(train_losses)
    loss_fn = 'loss_data.npy'
    np.save(os.path.join(save_path, loss_fn), losses_array)
