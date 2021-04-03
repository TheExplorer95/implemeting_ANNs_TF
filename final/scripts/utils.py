import time
import tensorflow as tf


def configure_gpu_options():
    # allows for GPU memory growth
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


class Timer:
    # A small class for making timings.
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


def check_dirs(path_list):
    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path)


def get_command_line_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_name",
                    type=str,
                    default='1dconv_gru',
                    help='Currently implemented: 1dconv_gru/, 1dconv_gru/,\
                    2dconv_gru/, 2dconv_transformer/')

    return vars(ap.parse_args())


def set_mixed_precission(bool):
    if bool:
        tf.keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )  # use mixed precision training (on V100 supposedly a 3x performance boost + double memory)
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
