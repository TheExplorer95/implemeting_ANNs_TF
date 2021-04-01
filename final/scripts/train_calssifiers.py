import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from params import *
from classifier_model import get_classifier
from generate_embeddings import create_classifier_dataset


def plot_classifier_training(history, epochs, save_plot_as):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(save_plot_as, bbox_inches='tight')


# 3 design principles
model = get_classifier(c_dim, 10)
optimizer = tf.keras.optimizers.Adam(learning_rate_class)
cce = tf.keras.losses.CategoricalCrossentropy()

# generate dataset from saved embeddings
train_ds = create_classifier_dataset(path_load_embeddings_train)
test_ds = create_classifier_dataset(path_load_embeddings_test)

# create save_plot_as list of filenames/paths for the plots

# iterate over datasets list and do the following. also iterate (with zip) over the save_plot_as list.

# train and test the model
model.compile(optimizer = optimizer,
              loss = cce,
              metrics=["accuracy"])
history = model.fit(train_ds, epochs = epochs_class, batch_size = batch_size_classifier,
                    validation_data = test_ds) # add additional arguments

### save history and plots
plot_classifier_training(history, epochs_class, plotname)
np.save(plotname.replace('png', 'npy'), history.history)
