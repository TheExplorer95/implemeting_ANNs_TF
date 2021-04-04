from params import *
import os
import random
import tensorflow_io as tfio
import numpy as np
import scipy

def decode_audio(audio_path, original_sr, desired_sr, duration, max_duration=30):
    """
    Loads and decodes wav file and applies sub- or supersampling to achieve a desired sampling rate.
    Pads the audio tensor with zeros up to max_duration and then randomly takes a duration seconds long random crop.
    """

    audio_binary = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(
        audio_binary, desired_channels=1, desired_samples=max_duration * original_sr
    )
    audio = tf.image.random_crop(audio, size=(duration * sr, 1))

    if not desired_sr == original_sr:
        audio = tfio.audio.resample(audio, original_sr, desired_sr)

    return tf.squeeze(audio, axis=-1), desired_sr


def preprocess_mel_spec(mel_spec):
    """
    Preprocesses single mel spectra: 1. Standardization along time axis,
    2. gaussian_filter along time axis.

    mel_spec: tf.tensor or numpy array with n_mels x n_timesteps dimensions
    """
    # standardization (over timesteps/per mel bin)
    mean = tf.expand_dims(tf.math.reduce_mean(mel_spec, axis=-1), axis=-1)
    std = tf.expand_dims(tf.math.reduce_std(mel_spec, axis=-1), axis=-1)
    mel_spec = (mel_spec - mean) / std

    # gaussian filter
    mel_spec = scipy.ndimage.gaussian_filter1d(mel_spec, sigma=2, axis=1)

    return tf.expand_dims(mel_spec, axis=-1)


def batch_data_generator():
    """
    Relies on a global argument dictionary "data_generator_arguments" containing the keys:

    T:                  Number of time-steps (each being an audio window) used for prediction
    k:                  Number of time-steps (each being an audio window) to predict
    N:                  Number of negative samples (false/random prediction audio-windows)
    original_sr:        Sampling rate of the audio files
    desired_sr:         Sampling rate used for resampling the audio files (can reduce computational load but cuts high frequencies)
    full_duration:      Length of audio files (shorter files get padded, longer files get cropped)
    data_path:          str, name of folder which contains multiple .wav data

    Negative samples are drawn only from other audio files in the batch as in [van den Oord et al 2018]. Batch size equals N.

    Outputs a batch tensor of shape (batch_size, T +k*N, window_size, 1)
    """

    T = data_generator_arguments["T"]
    k = data_generator_arguments["k"]
    N = data_generator_arguments["N"]
    original_sr = data_generator_arguments["original_sr"]
    desired_sr = data_generator_arguments["desired_sr"]
    duration = data_generator_arguments["full_duration"]
    data_path = data_generator_arguments["data_path"]
    batch_size = N

    window_size = duration * desired_sr / (T + k)
    assert (
        not window_size % 1
    ), f"duration*sample rate and (T+k) must be divisible. Currently duration*sample_rate = {duration * desired_sr} and (T+k) = {T + k}"
    window_size = int(window_size)
    folder = os.listdir(data_path)
    filepaths = [os.path.join(data_path, f) for f in folder]
    while True:

        # get audio from randomly sampled paths, truncated to duration and resampled to desired sr
        paths = random.sample(filepaths, batch_size)
        songs = [
            decode_audio(path, original_sr, desired_sr, duration)[0] for path in paths
        ]

        batch = []
        for idx in range(batch_size):
            samples = []
            positive_sample = songs[idx]
            positive_sample = tf.reshape(positive_sample, (1, T + k, window_size, 1))
            samples.append(positive_sample)

            # add a set of negative (not coming from index idx) sample audio windows of size (1,k,window_size,1)
            for i, audio in enumerate(songs):
                if i != idx:
                    samples.append(
                        tf.reshape(
                            tensor=tf.image.random_crop(audio, size=[window_size * k]),
                            shape=(1, k, window_size, 1),
                        )
                    )

            # get one sample with shape (1, T +k*N, window_size, 1)
            batch.append(tf.concat(samples, axis=1))

        yield tf.concat(batch, axis=0)  # yield complete batch from single
        # samples


def batch_data_generator_spectogram():
    """
    Relies on a global argument dictionary "data_generator_arguments" containing the keys:

    T:                  Number of time-steps (each being an audio window) used for prediction
    k:                  Number of time-steps (each being an audio window) to predict
    N:                  Number of negative samples (false/random prediction audio-windows)
    batch_size:         Batch size used for training the model (for 1dConv fixed: batch_size = N)
    data_path:          Path to the trainign data.

    Negative samples are drawn only from other audio files in the batch as in [van den Oord et al 2018].

    Outputs a batch tensor of shape (batch_size, T +k*N, window_size, 1)
    """
    # get dataset stuff
    data_path = data_generator_arguments["data_path"]
    batch_size = data_generator_arguments["batch_size"]
    data_fns = os.listdir(data_path)
    data_paths = np.array([os.path.join(data_path, fn) for fn in data_fns])
    sample = np.load(data_paths[0])

    # set window_size
    T = data_generator_arguments["T"]
    k = data_generator_arguments["k"]
    N = data_generator_arguments["N"]
    n_mels = sample.shape[0]
    window_size = n_mels  # assumes squared input
    n_samples = len(data_paths)

    while True:
        # 1. load and preprocess
        # random indices for batch amount of mel specs
        batch_idxs = tf.random.uniform(
            (batch_size,), minval=0, maxval=n_samples - 1, dtype=tf.dtypes.int32
        )

        mel_specs = [
            preprocess_mel_spec(np.load(data_paths[idx]))
            for i, idx in enumerate(batch_idxs)
        ]

        neg_img_idxs = tf.random.uniform(
            (batch_size, N - 1), minval=0, maxval=batch_size, dtype=tf.dtypes.int32
        )

        batch = []
        for batch_idx in tf.range(batch_size):
            samples = []
            # 2. get T timesteps and the k pos timesteps for f_score
            samples.extend(
                tf.split(
                    tf.image.random_crop(
                        mel_specs[batch_idx], size=[n_mels, window_size * (T + k), 1]
                    ),
                    num_or_size_splits=T + k,
                    axis=1,
                )
            )

            # 3. get k*N negative timesteps
            for neg_idx in neg_img_idxs[batch_idx]:
                samples.extend(
                    tf.split(
                        tf.image.random_crop(
                            mel_specs[neg_idx], size=[n_mels, window_size * k, 1]
                        ),
                        num_or_size_splits=k,
                        axis=1,
                    )
                )

            batch.append(tf.stack(samples))

        batch = tf.stack(batch)
        yield batch


def create_cpc_ds():
    """
    Uses a global dictionary "data_generator_arguments" to create a tf dataset from a generator that outputs batches already.

    The data_generator_arguments dictionary has the following arguments:

    T:                  Number of time-steps (each being an audio window) used for prediction
    k:                  Number of time-steps (each being an audio window) to predict
    N:                  Number of negative samples (false/random prediction audio-windows)
    original_sr:        Sampling rate of the audio files
    desired_sr:         Sampling rate used for resampling the audio files (can reduce computational load but cuts high frequencies)
    full_duration:      Length of audio files (shorter files get padded, longer files get cropped)
    batch_size:         Batch size used for training the model (for 1dConv fixed: batch_size = N)
    data_path:          Path to the trainign data.
    """

    if enc_model == "1d_conv":
        T = data_generator_arguments["T"]
        k = data_generator_arguments["k"]
        N = data_generator_arguments["N"]
        sampling_rate = data_generator_arguments["desired_sr"]
        batch_size = N
        duration = data_generator_arguments["full_duration"]
        sr = data_generator_arguments["desired_sr"]

        # output shape of generator given the arguments
        data_shape = (batch_size, T + k * N, int((duration * sr) / (T + k)), 1)

        train_ds = tf.data.Dataset.from_generator(
            generator=batch_data_generator,
            output_signature=tf.TensorSpec(
                data_shape, dtype=tf.dtypes.float32, name=None
            ),
        )

    elif enc_model == "2d_conv":
        T = data_generator_arguments["T"]
        k = data_generator_arguments["k"]
        N = data_generator_arguments["N"]
        batch_size = data_generator_arguments["batch_size"]
        sample_file_path = os.path.join(
            data_generator_arguments["data_path"],
            os.listdir(data_generator_arguments["data_path"])[0],
        )

        sample = np.load(sample_file_path)
        n_mels = sample.shape[0]
        window_size = n_mels  # assumes square input

        # output shape of generator given the arguments
        data_shape = (batch_size, T + k * N, n_mels, window_size, 1)

        train_ds = tf.data.Dataset.from_generator(
            generator=batch_data_generator_spectogram,
            output_signature=tf.TensorSpec(
                data_shape, dtype=tf.dtypes.float32, name=None
            ),
        )
    else:
        print(f"[Error] - The desired encoder model: {enc_model} was not implemented")

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds


def create_classifier_dataset(embedding_path):
    """
    Works only when the contained string doesn't have multiple class names

    Output: Create a tf dataset with labels
    """
    em_files = os.listdir(embedding_path)
    em_filepaths = [
        os.path.join(embedding_path, f) for f in em_files
    ]  # train files was created for training

    embedding_data = [np.reshape(np.load(x), (1, c_dim)) for x in em_filepaths]

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

    em_onehot_labels = [
        tf.reshape(tf.eye(len(classes))[l], (1, len(classes)))
        for l in [
            [i for i, label in enumerate(classes) if label in p][0]
            for p in em_filepaths
        ]
    ]

    ds = tf.data.Dataset.from_tensor_slices((embedding_data, em_onehot_labels))

    return ds.prefetch(tf.data.AUTOTUNE)


def create_autoencoder_dataset(embedding_path):
    embedding_data = []
    for path in embedding_path:  # use both train and test data
        em_files = os.listdir(path)
        em_filepaths = [
            os.path.join(path, f) for f in em_files
        ]  # train files was created for training
        embedding_data.append([np.reshape(np.load(x), (1, c_dim)) for x in em_filepaths])
    embedding_data = [item for sublist in embedding_data for item in sublist] # flatten lists
    ds = tf.data.Dataset.from_tensor_slices((embedding_data, embedding_data))


    return ds.prefetch(tf.data.AUTOTUNE)
