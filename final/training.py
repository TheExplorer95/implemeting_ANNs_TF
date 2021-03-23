import tensorflow as tf

#@tf.function
def train_step(model, ds, loss_function, optimizer,
               steps_per_epoch, train_loss_metric=None,
               train_acc_metric= None):
    '''
    Training for one epoch.
    '''

    for batch in ds.take(steps_per_epoch): # use 100 batch per epoch
        # forward pass with GradientTape
        with tf.GradientTape() as tape:
            prediction = model(batch)
            loss = loss_function(prediction)

        # backward pass via GradienTape (auto-gradient calc)
        if not tf.math.is_nan(loss):
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # update metrics
            if train_loss_metric is not None:
                train_loss_metric.update_state(loss)
            if train_acc_metric is not None:
                train_acc_metric.update_state(target, prediction)
        else:
            tf.print("loss is nan, no parameters updated")
            tf.print("f_matrix:", prediction)
