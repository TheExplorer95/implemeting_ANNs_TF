import tensorflow as tf

# use mixed precision in case of using big models
mixed_precision = False
if mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')  # use mixed precision training (on V100 supposedly a 3x performance boost + double memory)
else:
    tf.keras.mixed_precision.set_global_policy('float32')

# data path
modelname = '1dconv_gru/'  # '1dconv_gru', '2dconv_gru', '2dconv_transformer'
cwd = '/content/final'  # main path
# location of raw .wav data
path_data_train = cwd + '/GTZAN/'
path_data_test = cwd + '/test_data/'
# location to load intermediate weights if training wasn't done until the end
path_to_continue_training = False
# location to save weights and loss results from cpc training
path_save_cpc = cwd + '/results/cpc/' + modelname + 'weights.h5'
# location to load saved cpc model weights
path_load_model = cwd + '/results/cpc/' + modelname + 'weights.h5'
# location to save generated embeddings using trained cpc
path_save_embeddings_train = cwd + '/results/embeddings/' + modelname + 'train/'
path_save_embeddings_test = cwd + '/results/embeddings/' + modelname + 'test/'
# location to load saved embeddings
path_load_embeddings_train = cwd + '/results/embeddings/' + modelname + 'train/'
path_load_embeddings_test = cwd + '/results/embeddings/' + modelname + 'test/'
# location to save figures and history of classifier
plotname = cwd + '/results/classifier/' + modelname + 'plot.png'

# CPC data params
data_generator_arguments = {
    "T": 27,  # timestep
    "k": 3,  # timestep
    "N": 8,  # number
    "full_duration": 4,  # sec
    "original_sr": 22050,  # Hz
    "desired_sr": 4410,  # Hz
    "folder_path": path_data_train
    }

# TODO: Janosch, spectogram args
# data_generator_arguments = {
#     'num_mels'
# }

# classifier data params
batch_size_classifier = 32
num_em_samples_per_data = 10  # How often to sample from a single data to get different parts

# encoder params
z_dim = 256  # output dim
if modelname == '1dconv_transformer/' or '1dconv_gru/':
    enc_model = '1d_conv'
    encoder_args = {
        "z_dim": z_dim,
        "stride_sizes": [5, 4, 2, 2, 2],
        "kernel_sizes": [10, 8, 4, 4, 4],
        "n_filters": [512, 512, 512, 512, 512],
        "activation": tf.nn.leaky_relu
    }
elif modelname == '2dconv_gru/' or '2dconv_transformer/':
    enc_model = '2d_conv'
    # # mel_specto
    # encoder_args = {
    #     ''
    # }


# AR params
c_dim = 512

elif modelname == '1dconv_gru/' or '2dconv_gru/':
    ar_model = 'GRU'
    ar_args = {}
    
elif modelname == '1dconv_transformer/' or '2dconv_transformer/':
    ar_model = 'transformer'
    ar_args = {
        'num_enc_layers': 2,  # num. transformer encoder blocks
        'num_heads': 2,  # num. multiheads for attention
        'z_dim': z_dim,
        'dff': 100,  # num. units for 1st ffn within encoder block
        'dense_units': [100, c_dim],  # num. units for additional ffn
        'activation': tf.nn.leaky_relu,  # activation for additional ffn
        'maximum_position_encoding': data_generator_arguments['T'],
        'rate': 0.1  # dropout rate
    }

# training params
epochs_cpc = 1  #500
steps_per_epoch_cpc = 1  #100
epochs_class = 1  #1000
learning_rate = 2e-5  # for cpc
learning_rate_class = 1e-3  # for classifier
