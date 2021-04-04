from params import *

import numpy as np
import tensorflow as tf
import os

from classifier_model import get_classifier, DimensionalityReduction
from analysis import plot_classifier_training, plot_confusion_matrix
from preprocess_data import create_classifier_dataset, create_autoencoder_dataset
from utils import configure_gpu_options

if set_memory_growth_tf:
    configure_gpu_options()


# -------------- Training ----------------------------------------------
# 1. train the autoencoder for dimension reduction
em_ds = create_autoencoder_dataset(
    [path_load_embeddings_train, path_load_embeddings_test]
)  # dataset


# define tf functional api autoencoder model
reducer = DimensionalityReduction(c_dim, r_dim)  # model
mse = tf.keras.losses.MeanSquaredError()  # loss
reducer.compile(optimizer=optimizer_dimension_reduction, loss=mse)  # compile
# train autoencoder
history_dm = reducer.fit(
    em_ds,
    epochs=epochs_dimension_reduction,
    shuffle=True,
    batch_size=batch_size_dimension_reduction,
)
np.save(
    os.path.join(path_save_classifier_plots, "autoencoder_results.npy"),
    history_dm.history,
)
print("[INFO] - Autoencoder is trained")

# 2. generate dataset from saved embeddings
print("[INFO] - Creating the dataset for the train and test cpc-embeddings.")
train_ds = create_classifier_dataset(path_load_embeddings_train)
test_ds = create_classifier_dataset(path_load_embeddings_test)

# 3. 3 design principles
print("[INFO] - Initializing the classifier.")
classifier = get_classifier(c_dim, 10, reducer)  # use reducer to reduce dim beforehand
cce = tf.keras.losses.CategoricalCrossentropy()

# 4. train and test the classifier
classifier.compile(optimizer=optimizer_class, loss=cce, metrics=["accuracy"])
print(f"\n[INFO] - Created the {classifier.name} model.")

classifier.layers[1].trainable = False # do not train autoencoder in classifier
classifier.summary()
print(f"Inside the classifier, the Encoder of the Autoencoder is set to: trainable = {classifier.layers[1].trainable == True}")

print(f"\n[Info] - Start training the classifier.")
history = classifier.fit(
    train_ds,
    epochs=epochs_class,
    batch_size=batch_size_class,
    validation_data=test_ds.take(test_size_classifier),
)


# --------------- Analysis of the classifier ----------------------------------
# 1. Loss and accuracy plots
plot_classifier_training(history, epochs_class, path_save_classifier_plots)
exp_data_fn = "train_results.npy"
np.save(os.path.join(path_save_classifier_plots, exp_data_fn), history.history)

# 2. Confusion matrix
plot_confusion_matrix(
    test_ds, classifier, path_save_classifier_plots
)
