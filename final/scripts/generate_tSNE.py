from params import *
import numpy as np
from analysis import plot_tsne


def load_embeddings(path):
    """
    Used to load saved embeddings.
    :param path: str, path to where embeddings are saved
    :return: np.array, embeddings with dim (num_embeddings, c_dim)
    """
    em_files = os.listdir(path)
    em_filepaths = [os.path.join(path, f) for f in em_files]  # list of all file paths
    return np.concatenate(
        [np.reshape(np.load(x), (1, c_dim)) for x in em_filepaths], axis=0
    )


# TSNE plotting of the train and test embeddings
print("[INFO] - Do the TSNE.")

# load the data
embeddings_train = load_embeddings(path_load_embeddings_train)
embeddings_test = load_embeddings(path_load_embeddings_test)

# assign labels for each embedding
classes = [
    "blues",
    "reggae",
    "metal",
    "rock",
    "pop",
    "classical",
    "country",
    "disco",
    "jazz",
    "hiphop",
]

labels_train = np.array(
    [  # label of each data point in str
        [classes[i] for i, label in enumerate(classes) if label in p][0]
        for p in os.listdir(path_load_embeddings_train)
    ]
)

labels_test = np.array(
    [
        [classes[i] for i, label in enumerate(classes) if label in p][0]
        for p in os.listdir(path_load_embeddings_test)
    ]
)


# create 12 figures with a single tSNE on merged train and test data
plot_tsne(
    embeddings_train[:num_tsne],  # slice to take only first num_tsne data points
    embeddings_test[:num_tsne],
    labels_train[:num_tsne],
    labels_test[:num_tsne],
    path_save_classifier_plots,
    classes,
)
