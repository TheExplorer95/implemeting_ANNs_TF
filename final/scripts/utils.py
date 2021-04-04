import time
import os
import argparse
import tensorflow as tf


# Allows for GPU memory growth
def configure_gpu_options():

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception:
        print("[INFO] - Cannot activate memory growth.")


# Class to measure times
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        # Start a new timer
        self._start_time = time.perf_counter()

    def stop(self):
        # Stop the timer, and report the elapsed time
        if self._start_time is None:
            print("Timer is not running. Use .start() to start it")
            return 0

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time


# Create the appropriate directory structure to save results and figures from experiments
def check_dirs(path_list):
    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path)


# Used to parse script arguments
def get_command_line_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="1dconv_gru",
        help="Currently implemented: 1dconv_gru/, 1dconv_gru/,\
                    2dconv_gru/, 2dconv_transformer/",
    )

    return vars(ap.parse_args())


# Allow to use mixed precision training which can boost performance
def set_mixed_precission(bool):
    if bool:
        tf.keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )  # use mixed precision training (on V100 supposedly a 3x performance boost + double memory)
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
