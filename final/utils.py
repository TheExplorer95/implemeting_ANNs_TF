from tensorflow import compat

def configure_gpu_options():
    config = compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = compat.v1.Session(config=config)
    compat.v1.keras.backend.set_session(sess)


if __name__=='__main__':
    en = configure_gpu_options()
    print("done")
