import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
# f(x) = a(b(c(d(x))))
# function = [d, c, b, a]
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
_ = tf.compat.v1.Session(config=config)


# Load the encoded protein sequences, labels and masks
# The masks are not needed for the FFN or CNN models
train = np.load('data/reduced_train.npz')
X_train = train['X_train']
y_train = train['y_train']
mask_train = train['mask_train']
print(X_train.shape)
print(X_train)

validation = np.load('data/reduced_val.npz')
X_val = validation['X_val']
y_val = validation['y_val']
mask_val = validation['mask_val']
print(X_val.shape)

# Building the network by defining the network architecture.
# We have an input layer, dense layer and output layer.

# Set the hyperparameters of the network:
batch_size = 128
seq_len = 400
n_feat = 20
n_hid = 30
n_class = 10
lr = 0.0025
drop_prob = 0.5

# Dummy data to check the size of the layers during the building of the network
X = np.random.randint(0, 10, size=(batch_size, seq_len, n_feat))
# print("x: {}".format(X))

# Define the layers of the network
input_shape = (seq_len, n_feat)
model = Sequential()
# Input layer, holds the shape of the data, flattening the input
model.add(Flatten(input_shape=input_shape))
# # Dense layer with ReLu activation function
model.add(Dense(units=n_hid, activation='relu'))
model.add(Dropout(drop_prob))
# Output layer with a Softmax activation function
model.add(Dense(units=n_class, activation='softmax'))

# Calculate the prediction and network loss for the training set and update the network weights:

# todo c'è qualcosa che manca forse qui, ovvero:
# Training loss
# loss = T.mean(t_loss)
# Parameters
# params = lasagne.layers.get_all_params([l_out], trainable=True)
# Get the network gradients and perform total norm constraint normalization
# all_grads = lasagne.updates.total_norm_constraint(T.grad(loss, params),3)
# total_norm_constraint() constrain the total norm of a list of tensors e questo non viene fatto.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=lr),
              metrics=['accuracy'])

print(model.summary())
plot_model(model, "model.png", show_shapes=True)

# Train
y_train = to_categorical(y_train, n_class)
y_val = to_categorical(y_val, n_class)

# Number of epochs
num_epochs = 80

# Calculate also the prediction and network loss for the validation set:
history = model.fit(X_train, y_train, epochs=80, batch_size=batch_size, validation_data=(X_val, y_val), shuffle=True)

# Model loss and accuracy
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

# Confusion matrix
# The confusion matrix shows how well is predicted each class and which are the most common mis-classifications.
# Code based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# todo
