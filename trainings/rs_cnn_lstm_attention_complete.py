import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import talos
import os
from utils.models import CustomModels

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
_ = tf.compat.v1.Session(config=config)

# Load the encoded protein sequences and labels
train = np.load('./data/psi-blast/1000_train_val.npz')
X_train_val = train['X_train']
y_location = train['y_train_location']
y_membrane = train['y_train_membrane']

# Split train set from validation set
val_percentage = 20
train_val_split_point = X_train_val.shape[0] - int((X_train_val.shape[0]*val_percentage)/100)
X_train, X_val = np.split(X_train_val, [train_val_split_point])
y_train_location, y_val_location = np.split(y_location, [train_val_split_point])
y_train_membrane, y_val_membrane = np.split(y_membrane, [train_val_split_point])

# Hyperparameters to check
p = {
    'batch_size': (32, 256, 32),
    'lr': [0.0001, 0.0005, 0.001, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.004, 0.005, 0.007],
    'n_filt': (5, 50, 5),
    'n_hid': (5, 100, 5),
    'drop_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'drop_hid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
}

experiment_name = "random_search/CNN_LSTM_Attention"

n_class_location = 10
n_class_membrane = 2

# One-hot encoding
y_train_location = to_categorical(y_train_location, n_class_location)
y_train_membrane = to_categorical(y_train_membrane, n_class_membrane)
y_val_location = to_categorical(y_val_location, n_class_location)
y_val_membrane = to_categorical(y_val_membrane, n_class_membrane)

# Load model
custom_model = CustomModels(seq_len=1000, n_feat=20, n_hid=p['n_hid'], n_class=10, lr=p['lr'], drop_prob=p['drop_prob'],
                            n_filt=p['n_filt'], random_search=True, n_membrane_class=n_class_membrane,
                            batch_size=p['batch_size'])

# Random search
t = talos.Scan(x=X_train, y=[y_train_location, y_train_membrane], x_val=X_val, y_val=[y_val_location, y_val_membrane],
               params=p, fraction_limit=0.0005, model=custom_model.create_CNN_LSTM_Attention_complete, performance_target='val_subcellular_loss',
               experiment_name=experiment_name, disable_progress_bar=False, print_params=True)
