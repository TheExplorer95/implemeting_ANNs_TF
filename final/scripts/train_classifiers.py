from params import *

import numpy as np
import tensorflow as tf
import os

from classifier_model import get_classifier
from analysis import plot_classifier_training, plot_confusion_matrix
from preprocess_data import create_classifier_dataset
from utils import configure_gpu_options
if set_memory_growth_tf:
    configure_gpu_options()


# -------------- Training ----------------------------------------------
# 1. 3 design principles
print('[INFO] - Initializing the classifier.')
classifier = get_classifier(c_dim, 10)
optimizer = tf.keras.optimizers.Adam(learning_rate_class)
cce = tf.keras.losses.CategoricalCrossentropy()

# 2. generate dataset from saved embeddings
print('[INFO] - Creating the dataset for the train and test cpc-embeddings.')
train_ds = create_classifier_dataset(path_load_embeddings_train)
test_ds = create_classifier_dataset(path_load_embeddings_test)

# 3. train and test the classifier
classifier.compile(optimizer=optimizer, loss=cce, metrics=["accuracy"])
print(f'\n[INFO] - Created the {classifier.name} model.')
classifier.summary()

print(f"\n[Info] - Start training the classifier.")
history = classifier.fit(
    train_ds,
    epochs=epochs_class,
    batch_size=batch_size_classifier,
    validation_data=test_ds,
)  # add additional arguments


# --------------- Analysis of the classifier ----------------------------------
# 1. Loss
plot_classifier_training(history, epochs_class, path_save_classifier_plots)
exp_data_fn = "train_results.npy"
np.save(os.path.join(path_save_classifier_plots, exp_data_fn), history.history)

# 2. Confusion matrix
plot_confusion_matrix(test_ds, classifier, path_save_classifier_plots)
