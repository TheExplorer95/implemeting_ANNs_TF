import tensorflow as tf
import numpy as np
import random
import librosa

data_generator_cpc_arguments = {
    "T" : 20,
    "k" : 10,
    "N" : 8,
    "window_duration" : 1,
    "full_duration" : 30,
    "original_sr" : 22050,
    "desired_sr" : 22050,
    "filepaths" : None,
    "batch_size" : 8
    }

def get_cpc_gen(args):
    global data_generator_cpc_arguments
    data_generator_cpc_arguments = args
    return data_generator_cpc

def decode_audio(audio_path, original_sr, desired_sr, duration):
    """decodes wav file and applies sub- or supersampling to achieve a desired sampling rate"""
    audio, sr = librosa.load(audio_path, sr=None)
    audio = audio[: sr*duration] # cut audio files to desired length
    if not len(audio) == sr*duration:
        mock = np.zeros(sr*duration-len(audio))
        audio = np.append(audio, mock)
    assert len(audio) == sr*duration
    audio = librosa.resample(audio, original_sr, desired_sr)
    return tf.constant(audio, dtype=tf.float32), desired_sr


def data_generator_cpc():
    """
    Relies on a global argument dictionary

    T: number of time-steps to use for the context embedding c_t

    k: number of future time-steps to predict

    N: number of samples (1 positive + N-1 negative samples)

    window_duration: window duration in seconds

    original_sr: sampling rate of original audio files

    desired_sr: sampling rate as input to CPC (used for subsampling)

    duration: assumed full duration of an audio file (30s, files get truncated or padded with zeros)

    filepaths: list of all filepaths to all audio files
    """
    global data_generator_cpc_arguments
    T = data_generator_cpc_arguments["T"]
    k = data_generator_cpc_arguments["k"]
    N = data_generator_cpc_arguments["N"]
    window_duration = data_generator_cpc_arguments["window_duration"]
    original_sr = data_generator_cpc_arguments["original_sr"]
    desired_sr = data_generator_cpc_arguments["desired_sr"]
    duration = data_generator_cpc_arguments["full_duration"]
    filepaths = data_generator_cpc_arguments["filepaths"]

    while True:

        #randomly select sample filepaths from list for N samples
        samples = random.sample(filepaths, N)
        positive_sample = samples[0]
        negative_samples = samples[1:]

        # take full 30 seconds of positive sample
        positive_audio, sample_rate = decode_audio(positive_sample, original_sr,
                                                   desired_sr, duration)
        window_size = int(sample_rate * window_duration)
        positive_audio = tf.reshape(positive_audio, (T+k, window_size, 1))

        # negative samples (find a way to do it without a for loop pls)
        sample_tensors = []
        sample_tensors.append(positive_audio)

        for ns in negative_samples:
#             ns = tf.io.read_file(ns)
            ns, sample_rate = decode_audio(ns, original_sr,
                                           desired_sr, duration)
            # cut to 30s
            window_size = int(sample_rate * window_duration)
            ns = tf.reshape(ns, (T+k,window_size,1))

            # only take the last k entries (better: a random part of the audio)
            ns = ns[T:T+k]
            sample_tensors.append(ns)

        # concatenate all tensors, making its shape (T+k*N, window_size,1)
        data = tf.concat(sample_tensors, axis= 0)

        yield data
