import os
import random
import tensorflow_io as tfio

from params import *


def decode_audio(audio_path, original_sr, desired_sr, duration, max_duration=30):
    """
    Loads and decodes wav file and applies sub- or supersampling to achieve a desired sampling rate.
    Pads the audio tensor with zeros up to max_duration and then randomly takes a duration seconds long random crop.
    """

    audio_binary = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=max_duration * original_sr)
    audio = tf.image.random_crop(audio, size=(duration * sr, 1))

    if not desired_sr == original_sr:
        audio = tfio.audio.resample(audio, original_sr, desired_sr)

    return tf.squeeze(audio, axis=-1), desired_sr


def batch_data_generator():
    """
    Relies on a global argument dictionary "data_generator_arguments" containing the keys:

    T:                  Number of time-steps (each being an audio window) used for prediction
    k:                  Number of time-steps (each being an audio window) to predict
    N:                  Number of negative samples (false/random prediction audio-windows)
    original_sr:        Sampling rate of the audio files
    desired_sr:         Sampling rate used for resampling the audio files (can reduce computational load but cuts high frequencies)
    full_duration:      Length of audio files (shorter files get padded, longer files get cropped)
    folder_path:        str, name of folder which contains multiple .wav data

    Negative samples are drawn only from other audio files in the batch as in [van den Oord et al 2018]. Batch size equals N.

    Outputs a batch tensor of shape (batch_size, T +k*N, window_size, 1)
    """

    T = data_generator_arguments["T"]
    k = data_generator_arguments["k"]
    N = data_generator_arguments["N"]
    original_sr = data_generator_arguments["original_sr"]
    desired_sr = data_generator_arguments["desired_sr"]
    duration = data_generator_arguments["full_duration"]
    folder_path = data_generator_arguments["folder_path"]
    batch_size = N

    window_size = duration * desired_sr / (T + k)
    assert not window_size % 1, f"duration*sample rate and (T+k) must be divisible. Currently duration*sample_rate = {duration * desired_sr} and (T+k) = {T + k}"
    window_size = int(window_size)
    folder = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, f) for f in folder]
    while True:

        # get audio from randomly sampled paths, truncated to duration and resampled to desired sr
        paths = random.sample(filepaths, batch_size)
        songs = [decode_audio(path, original_sr, desired_sr, duration)[0] for path in paths]

        batch = []
        for idx in range(batch_size):
            samples = []
            positive_sample = songs[idx]
            positive_sample = tf.reshape(positive_sample, (1, T + k, window_size, 1))
            samples.append(positive_sample)

            # add a set of negative (not coming from index idx) sample audio windows of size (1,k,window_size,1)
            for i, audio in enumerate(songs):
                if i != idx:
                    samples.append(tf.reshape(
                        tensor=tf.image.random_crop(audio, size=[window_size * k]),
                        shape=(1, k, window_size, 1)))

            # get one sample with shape (1, T +k*N, window_size, 1)
            batch.append(tf.concat(samples, axis=1))

        yield tf.concat(batch, axis=0)  # yield complete batch from single samples


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
    filepaths:          List of filepaths to wav files.
    """

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
        output_signature=tf.TensorSpec(data_shape,
                                       dtype=tf.dtypes.float32,
                                       name=None)
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE).cache()

    return train_ds


# TODO: Janosch, add data_gen for spectogram
