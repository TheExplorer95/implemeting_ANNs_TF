import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from params import *
from cpc_model import CPC, Predict_z

### from a folder with embedding npy files, create a tf dataset with labels
def create_classifier_dataset(embedding_path):
    '''
    Works only when the contained string doesn't have multiple class names
    '''
    em_files = os.listdir(embedding_path)
    em_filepaths = [os.path.join(embedding_path, f) for f in em_files] # train files was created for training

    embedding_data = [np.load(x) for x in em_filepaths]

    classes = ["blues", "reggae", "metal", "rock", "pop", "classical", "country", "disco", "jazz", "hiphop"]

    em_onehot_labels = [tf.eye(len(classes))[i for i, label in enumerate(classes) if label in p] for p in em_filepaths]

    ds = tf.data.Dataset.from_tensor_slices((embedding_data, em_onehot_labels))

    return ds


### create embeddings (requires its own main script where all trained models are used to get embeddings)
def generate_embeddings(model, num_em_samples_per_data, folder_path, save_to, max_duration=30):
    original_sr = data_generator_arguments['original_sr']
    desired_sr = data_generator_arguments['desired_sr']
    duration = data_generator_arguments['full_duration']
    segments = data_generator_arguments['T'] + data_generator_arguments['k']
    segment_length = int(duration*desired_sr/segments)

    folder = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, f) for f in folder]
    for i in range(num_em_samples_per_data):

        for fpath in filepaths:
            audio_binary = tf.io.read_file(fpath)
            audio, sr = tf.audio.decode_wav(audio_binary, desired_channels = 1, desired_samples = max_duration * original_sr)

            if not desired_sr == original_sr:
                audio = tfio.audio.resample(audio, original_sr, desired_sr)

            audio = tf.squeeze(audio, axis=-1)
            audio = tf.image.random_crop(audio, size = (segments * segment_length,))
            audio = tf.reshape(audio, (1,segments, segment_length, 1))

            embedding = model.get_embedding(audio)
            embedding = tf.squeeze(embedding, axis= 0)
            save_to = save_to + str(i) + os.path.basename(fpath).replace(".wav", ".npy")
            np.save(save_to, embedding.numpy())


# Load the trained model
# init
cpc = CPC(data_generator_arguments["T"], data_generator_arguments["k"], data_generator_arguments["N"],
          z_dim, c_dim, enc_model, ar_model, Predict_z, encoder_args, ar_args, mixed_precision)
# compile by feeding dummy data
T = data_generator_arguments["T"]
k = data_generator_arguments["k"]
N = data_generator_arguments["N"]
sampling_rate = data_generator_arguments["desired_sr"]
batch_size = N
duration = data_generator_arguments["full_duration"]
sr = data_generator_arguments["desired_sr"]
data_shape = (batch_size, T + k * N, int((duration * sr) / (T + k)), 1)
dummy_data = tf.random.normal(data_shape, 0, 1)
cpc(dummy_data)
cpc.summary()
# load trained model
if path_load_model:
    cpc.load_weights(path_load_model)

# Use CPC to generate embeddings
generate_embeddings(cpc, num_em_samples_per_data, path_data_train, path_save_embeddings_train)
generate_embeddings(cpc, num_em_samples_per_data, path_data_test, path_save_embeddings_test)
