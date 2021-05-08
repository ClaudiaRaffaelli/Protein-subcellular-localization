import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm

from keras import backend as K
from keras.models import Model
from keras.layers import *

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix

import math
import random


#### Build network ####

# Set hyperparameters
batch_size = 128
seq_len = 400
n_feat = 20
n_hid = 30
n_class = 10
lr = 0.0025
n_filt = 10
drop_prob = 0.5
num_epochs = 80

# Build model
inputs = keras.Input(shape=(seq_len, n_feat))

l_conv_a = layers.Conv1D(n_filt, 3, strides=1, padding="same", activation="relu")(inputs)
l_conv_a = K.permute_dimensions(l_conv_a, (0, 2, 1))

l_conv_b = layers.Conv1D(n_filt, 5, strides=1, padding="same", activation="relu")(inputs)
l_conv_b = K.permute_dimensions(l_conv_b, (0, 2, 1))

l_conc = tf.keras.layers.Concatenate(axis=1)([l_conv_a, l_conv_b])
l_conc = K.permute_dimensions(l_conc, (0, 2, 1))

l_conv_final = layers.Conv1D(n_filt*2, 3, strides=1, padding="same", activation="relu")(l_conc)
final_max_pool = layers.MaxPooling1D(5)(l_conv_final)
final_max_pool = K.permute_dimensions(final_max_pool, (0, 2, 1))

# final_max_pool = Reshape((-1,))(final_max_pool)
# final_max_pool = tf.reshape(final_max_pool, tf.stack([-1, K.prod(K.shape(final_max_pool)[1:])]))
flatten_shape = np.prod((final_max_pool.shape[1], final_max_pool.shape[2])) # value of shape in the non-batch dimension
final_max_pool = tf.reshape(final_max_pool, shape=[-1, flatten_shape])

l_dense = layers.Dense(n_hid, activation="relu")(final_max_pool)
l_dense = layers.Dropout(drop_prob)(l_dense)

l_out = layers.Dense(n_class, activation="softmax")(l_dense)

model = keras.Model(inputs, l_out)


#### Load dataset ####
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
_ = tf.compat.v1.Session(config=config)

# Load the encoded protein sequences, labels and masks
# The masks are not needed for the FFN or CNN models
train = np.load('data/reduced_train.npz')
X_train = train['X_train']
y_train = train['y_train']

validation = np.load('data/reduced_val.npz')
X_val = validation['X_val']
y_val = validation['y_val']

# Or load selfmade datasets (Not working)
# X_train, y_train, X_val, y_val = load_selfmade_datasets() # method defined in utilities

#### Train ####
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=lr),
              metrics=['accuracy'])

y_train_oh = to_categorical(y_train, n_class)
y_val_oh = to_categorical(y_val, n_class)

history = model.fit(X_train, y_train_oh, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val_oh), shuffle=True)


#### Plot ####
# Plots of loss and accuracy for training and validation set at each epoch
x_axis = range(num_epochs)
plt.figure(figsize=(8, 6))
# loss_training:
plt.plot(x_axis, history.history['loss'])
# loss_validation
plt.plot(x_axis, history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(('Training', 'Validation'))
plt.show()

plt.figure(figsize=(8, 6))
# accuracy training
plt.plot(x_axis, history.history['accuracy'])
# accuracy validation
plt.plot(x_axis, history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(('Training', 'Validation'))
plt.show()


#### Confusion Matrix ####
Y_pred = model.predict(X_val)
y_pred = np.argmax(Y_pred, axis=1)

confusion_mat = confusion_matrix(validation['y_val'], y_pred)

plt.figure(figsize=(8, 8))
colormap = plt.cm.Blues
plt.imshow(confusion_mat, interpolation='nearest', cmap=colormap)
plt.title('Confusion matrix validation set')
plt.colorbar()
tick_marks = np.arange(n_class)
classes = ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell membrane', 'ER', 'Chloroplast',
                'Golgi apparatus', 'Lysosome', 'Vacuole']

plt.xticks(tick_marks, classes, rotation=60)
plt.yticks(tick_marks, classes)

thresh = confusion_mat.max() / 2.
for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
    plt.text(j, i, confusion_mat[i, j],
             horizontalalignment="center",
             color="white" if confusion_mat[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True location')
plt.xlabel('Predicted location')
plt.show()