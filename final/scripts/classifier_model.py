import tensorflow as tf
import os
import numpy as np

# TODO: optimize the architecture of classifier
def get_classifier(c_dim, num_classes):
    embedding_inputs = tf.keras.Input(shape=(c_dim))
    x = tf.keras.layers.Dense(64, activation="relu")(embedding_inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=embedding_inputs, outputs=outputs, name="music_classifier")
    return model

### from a folder with embedding npy files, create a tf dataset with labels
def create_classifier_dataset(embedding_path):
    '''
    Works only when the contained string doesn't have multiple class names
    '''
    em_files = os.listdir(embedding_path)
    em_filepaths = [os.path.join(embedding_path, f) for f in em_files] # train files was created for training

    embedding_data = [np.load(x) for x in em_filepaths]

    classes = ["blues", "reggae", "metal", "rock", "pop", "classical", "country", "disco", "jazz", "hiphop"]

    em_onehot_labels = [tf.eye(len(classes))[l] for l in [[i for i, label in enumerate(classes) if label in p][0] for p in em_filepaths]]

    ds = tf.data.Dataset.from_tensor_slices((embedding_data, em_onehot_labels))

    return ds
