from params import *

import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from cpc_model import CPC, Predict_z
from analysis import plot_tsne
from preprocess_data import preprocess_mel_spec


def load_embeddings(path):
    em_files = os.listdir(path)
    em_filepaths = [
        os.path.join(path, f) for f in em_files
    ]  # train files was created for training
    return np.concatenate(
        [np.reshape(np.load(x), (1, c_dim)) for x in em_filepaths], axis=0
    )


### create embeddings (requires its own main script where all trained models are used to get embeddings)
def generate_embeddings(
    model, num_em_samples_per_data, folder_path, save_to, max_duration=30
):
    if enc_model == "1d_conv":
        original_sr = data_generator_arguments["original_sr"]
        desired_sr = data_generator_arguments["desired_sr"]
        duration = data_generator_arguments["full_duration"]
        segments = data_generator_arguments["T"] + data_generator_arguments["k"]
        segment_length = int(duration * desired_sr / segments)

        folder = os.listdir(folder_path)
        filepaths = [os.path.join(folder_path, f) for f in folder]
        n_files = len(filepaths)
        total_embeddings = n_files * num_em_samples_per_data

        counter = 0

        for fpath in filepaths:
            audio_binary = tf.io.read_file(fpath)
            audio, sr = tf.audio.decode_wav(
                audio_binary,
                desired_channels=1,
                desired_samples=max_duration * original_sr,
            )
            if not desired_sr == original_sr:
                audio = tfio.audio.resample(audio, original_sr, desired_sr)
            for i in range(num_em_samples_per_data):
                audio_load = tf.squeeze(audio, axis=-1)
                audio_load = tf.image.random_crop(audio_load, size=(segments * segment_length,))
                audio_load = tf.reshape(audio_load, (1, segments, segment_length, 1))
                if modelname.split("_")[-1] == "transformer/":
                    embedding = model.get_embedding(
                        audio_load[:, : data_generator_arguments["T"], :, :]
                    )
                else:
                    embedding = model.get_embedding(audio_load)

                save_to_ = (
                    save_to + str(i+61) + os.path.basename(fpath).replace(".wav", ".npy")
                )
                np.save(save_to_, embedding.numpy())
                counter += 1
                if counter % 100 == 0:
                    print(f"[INFO] - Embeddings generated: {counter}, Embeddings remaining: {total_embeddings-counter}")

    elif enc_model == "2d_conv":
        # output dim (1, 516) are save npy arrays
        folder = os.listdir(folder_path)
        filepaths = [os.path.join(folder_path, f) for f in folder]

        segments = data_generator_arguments["T"] + data_generator_arguments["k"]
        sample = np.load(filepaths[0])
        n_mels = sample.shape[0]
        segment_length = n_mels
        n_files = len(filepaths)
        total_embeddings = n_files * num_em_samples_per_data

        counter = 0
        for fpath in filepaths:
            for i in range(num_em_samples_per_data):
                # load the file
                mel_spec = tf.squeeze(preprocess_mel_spec(np.load(fpath)))
                # get random window as input for the encoder
                mel_spec = tf.image.random_crop(
                    mel_spec, size=(n_mels, segments * segment_length)
                )
                # make slices from entire windown
                mel_spec = tf.stack(
                    tf.split(mel_spec, num_or_size_splits=segments, axis=1)
                )
                # add batch and channel dim
                mel_spec = tf.expand_dims(
                    tf.expand_dims(mel_spec, axis=-1), axis=0
                )

                # save the embedding
                if modelname.split("_")[-1] == "transformer/":
                    embedding = model.get_embedding(
                        mel_spec[:, : data_generator_arguments["T"], :, :, :]
                    )
                else:  # gru
                    embedding = model.get_embedding(mel_spec)

                # save the embedding
                if not tf.reduce_any(tf.math.is_nan(embedding)) and not tf.reduce_any(tf.math.is_finite(embedding)):
                    print(f'[ERROR] - Can not extract embedding from mel_spec {counter}')
                    print(embedding)
                else:
                    save_to_ = save_to + str(i) + os.path.basename(fpath)
                    np.save(save_to_, embedding.numpy())

                    counter += 1
                    if counter % 100 == 0:
                        print(f"[INFO] - Embeddings generated: {counter}, Embeddings remaining: {total_embeddings-counter}")


# # Load the trained model
# # init
# print("[INFO] - Initializing the classifier.")
# cpc = CPC(
#     data_generator_arguments["T"],
#     data_generator_arguments["k"],
#     data_generator_arguments["N"],
#     z_dim,
#     c_dim,
#     enc_model,
#     ar_model,
#     Predict_z,
#     encoder_args,
#     ar_args,
#     mixed_precision,
# )
#
# if enc_model == "1d_conv":
#     # compile by feeding dummy data
#     T = data_generator_arguments["T"]
#     k = data_generator_arguments["k"]
#     N = data_generator_arguments["N"]
#     sampling_rate = data_generator_arguments["desired_sr"]
#     batch_size = N
#     duration = data_generator_arguments["full_duration"]
#     sr = data_generator_arguments["desired_sr"]
#     data_shape = (batch_size, T + k * N, int((duration * sr) / (T + k)), 1)
#     dummy_data = tf.random.normal(data_shape, 0, 1)
#     cpc(dummy_data)
#     cpc.summary()
#
# elif enc_model == "2d_conv":
#     # compile by feeding dummy data
#     T = data_generator_arguments["T"]
#     k = data_generator_arguments["k"]
#     N = data_generator_arguments["N"]
#     batch_size = data_generator_arguments["batch_size"]
#     sample_file_path = os.path.join(
#         data_generator_arguments["data_path"],
#         os.listdir(data_generator_arguments["data_path"])[0],
#     )
#
#     sample = np.load(sample_file_path)
#     n_mels = sample.shape[0]
#     window_size = n_mels  # assumes square input
#
#     # output shape of generator given the arguments
#     data_shape = (batch_size, T + k * N, n_mels, window_size, 1)
#
#     dummy_data = tf.random.normal(data_shape, 0, 1)
#     cpc(dummy_data)
#     cpc.summary()
#
# # load trained model
# cpc.load_weights(path_load_model)
#
# # Create the embeddings for train and test data
# print("[INFO] - Generating embeddings.")
# generate_embeddings(
#     cpc, num_em_samples_per_train_data, path_data_train, path_save_embeddings_train
# )
# generate_embeddings(
#     cpc, num_em_samples_per_test_data, path_data_test, path_save_embeddings_test
# )

# -------TSNE plotting of the train and test embeddings---------------------
print("[INFO] - Do the TSNE.")

# load the data


embeddings_train = load_embeddings(path_load_embeddings_train)
embeddings_test = load_embeddings(path_load_embeddings_test)

# compute the labels
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


# each genre with merged test and train data
plot_tsne(embeddings_train[:num_tsne],
                    embeddings_test[:num_tsne],
                    labels_train[:num_tsne],
                    labels_test[:num_tsne],
                    path_save_classifier_plots,
                    classes)
