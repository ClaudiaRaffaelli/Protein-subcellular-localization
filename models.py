import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow import keras
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix


class Attention(tf.keras.layers.Layer):
	""" Implementing a layer that does attention according to Bahdanau style """

	def __init__(self, units):
		super(Attention, self).__init__()
		# W1 weight of the previously hidden state(hidden_size x hidden_size)
		self.W1 = tf.keras.layers.Dense(units)
		# W2 weight for all the encoder hidden states
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, features, hidden):
		# 'hidden' (h_t) is expanded over the time axis to prepare it for the addition
		# that follows. hidden will always be the last hidden state of the RNN.
		# (in seq2seq in would have been the current state of the decoder step)
		# 'features' (h_s) are all the hidden states of the encoder.
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# Bahdanau additive style to compute the score:
		# score = v_a * tanh(W_1*h_t + W_2*h_s)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
		attention_weights = tf.nn.softmax(self.V(score), axis=1)
		context_vector = attention_weights * features
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


class CustomModels:

	def __init__(self, seq_len, n_feat, n_hid, n_class, lr, drop_prob, n_filt=None):
		"""
		Hyperparameters of the network:
		:param seq_len: length of sequence
		:param n_feat: number of features encoded
		:param n_hid: number of hidden neurons
		:param n_class: number of classes to output
		:param lr: learning rate
		:param drop_prob: hidden neurons dropout probability
		:param n_filt: (optional) filters number
		"""
		self.seq_len = seq_len
		self.n_feat = n_feat
		self.n_hid = n_hid
		self.n_class = n_class
		self.lr = lr
		self.drop_prob = drop_prob
		self.n_filt = n_filt
		self.model = None

	def create_FFN(self):
		"""
		Building the network by defining its architecture: input layer, dense layer, output layer
		"""

		# Define the layers of the network
		input_shape = (self.seq_len, self.n_feat)
		self.model = Sequential()
		# Input layer, holds the shape of the data, flattening the input
		self.model.add(layers.Flatten(input_shape=input_shape))
		# Dense layer with ReLu activation function
		self.model.add(layers.Dense(units=self.n_hid, activation='relu'))
		self.model.add(layers.Dropout(self.drop_prob))
		# Output layer with a Softmax activation function
		self.model.add(layers.Dense(units=self.n_class, activation='softmax'))

		# Calculate the prediction and network loss for the training set and update the network weights:
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])
		# todo weights normalization?

	def create_CNN(self):
		"""
		Building the network by defining its architecture: input layer, two convolutional layers with max pooling,
															a dense layer and an output layer.
		"""

		# Build model
		inputs = keras.Input(shape=(self.seq_len, self.n_feat))
		l_permute = layers.Permute((2, 1))(inputs)

		l_conv_a = layers.Conv1D(self.n_filt, 3, strides=1, padding="same", activation="relu", data_format='channels_first') \
			(l_permute)
		l_conv_b = layers.Conv1D(self.n_filt, 5, strides=1, padding="same", activation="relu", data_format='channels_first') \
			(l_permute)
		l_conc = tf.keras.layers.Concatenate(axis=1)([l_conv_a, l_conv_b])

		l_conv_final = layers.Conv1D(self.n_filt * 2, 3, strides=1, padding="same", activation="relu",
									 data_format='channels_first')(l_conc)
		l_reshu = layers.Permute((2, 1))(l_conv_final)

		final_max_pool = layers.MaxPooling1D(5)(l_reshu)
		final_max_pool = layers.Flatten()(final_max_pool)

		l_dense = layers.Dense(self.n_hid, activation="relu")(final_max_pool)
		l_dense = layers.Dropout(self.drop_prob)(l_dense)
		l_out = layers.Dense(self.n_class, activation="softmax")(l_dense)

		self.model = keras.Model(inputs, l_out)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])


	def create_CNN_LSTM(self):
		"""
		Building the network by defining its architecture: input layer, two convolutional layers, a bidirectional LSTM,
															a dense layer and an output layer
		"""

		# Build model defining the layers
		# Define input
		l_input = keras.Input(shape=(self.seq_len, self.n_feat))

		l_permute = layers.Permute((2, 1))(l_input)

		# Convolutional layers with filter size 3 and 5
		l_conv_a = layers.Conv1D(self.n_filt, 3, strides=1, padding="same", activation="relu", data_format='channels_first')(
			l_permute)

		l_conv_b = layers.Conv1D(self.n_filt, 5, strides=1, padding="same", activation="relu", data_format='channels_first')(
			l_permute)

		# The output of the two convolution is concatenated
		l_conc = tf.keras.layers.Concatenate(axis=1)([l_conv_a, l_conv_b])

		# Building a second CNN layer
		l_conv_final = layers.Conv1D(
			self.n_filt * 2, 3, strides=1, padding="same", activation="relu", data_format='channels_first')(l_conc)

		# Second permute layer
		l_reshu = layers.Permute((2, 1))(l_conv_final)

		# mask to ignore the padded positions
		# todo mask here?
		l_mask = layers.Masking(mask_value=0.0)(l_reshu)

		# Bidirectional LSTM layer, taking only the last hidden state (only_return_final)
		l_fwd = layers.LSTM(units=self.n_hid, activation="tanh", return_sequences=False)(l_mask)
		l_bwd = layers.LSTM(units=self.n_hid, activation="tanh", return_sequences=False, go_backwards=True)(l_mask)

		# Concatenate both layers
		l_conc_lstm = tf.keras.layers.Concatenate(axis=1)([l_fwd, l_bwd])

		# Dense layer with ReLu activation function
		l_dense = layers.Dense(self.n_hid * 2, activation="relu")(l_conc_lstm)

		# Output layer with a Softmax activation function. Note that we include a dropout layer
		l_dropout = layers.Dropout(self.drop_prob)(l_dense)
		l_out = layers.Dense(self.n_class, activation="softmax")(l_dropout)

		self.model = keras.Model(l_input, l_out)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])

	def create_CNN_LSTM_Attention(self):
		"""
		Building the network by defining its architecture: an input layer, two convolutional layers, a bidirectional
														LSTM, an attention layer, a dense layer and an output layer.
		"""

		# Build model
		inputs = keras.Input(shape=(self.seq_len, self.n_feat))

		# todo mask here?
		# mask to ignore the padded positions
		l_mask = layers.Masking(mask_value=0.0, input_shape=(None, self.n_feat))(inputs)

		l_permute = layers.Permute((2, 1))(l_mask)

		l_conv_a = layers.Conv1D(self.n_filt, 3, strides=1, padding="same", activation="relu", data_format='channels_first')(
			l_permute)
		l_conv_b = layers.Conv1D(self.n_filt, 5, strides=1, padding="same", activation="relu", data_format='channels_first')(
			l_permute)
		l_conc = tf.keras.layers.Concatenate(axis=1)([l_conv_a, l_conv_b])

		l_conv_final = layers.Conv1D(
			self.n_filt * 2, 3, strides=1, padding="same", activation="relu", data_format='channels_first')(l_conc)

		l_reshu = layers.Permute((2, 1))(l_conv_final)

		# encoders LSTM
		l_lstm, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional \
			(layers.LSTM(self.n_hid, dropout=0.2, return_sequences=True, return_state=True, activation="tanh"))(l_reshu)

		state_h = layers.Concatenate()([forward_h, backward_h])
		state_c = layers.Concatenate()([forward_c, backward_c])

		# Set up the attention layer
		context_vector, attention_weights = Attention(self.n_hid * 2)(l_lstm, state_h)

		l_drop = layers.Dropout(self.drop_prob)(context_vector)

		l_out = layers.Dense(self.n_class, activation="softmax")(l_drop)

		self.model = keras.Model(inputs, l_out)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])
		# todo weights normalization?

	def confusion_matrix(self, X_val, validation):
		# The confusion matrix shows how well is predicted each class and which are the most common mis-classifications.
		Y_pred = self.model.predict(X_val)
		y_pred = np.argmax(Y_pred, axis=1)

		confusion_mat = confusion_matrix(validation['y_val'], y_pred)

		plt.figure(figsize=(8, 8))
		colormap = plt.cm.Blues
		plt.imshow(confusion_mat, interpolation='nearest', cmap=colormap)
		plt.title('Confusion matrix validation set')
		plt.colorbar()
		tick_marks = np.arange(self.n_class)
		classes = ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell membrane', 'ER', 'Chloroplast',
				   'Golgi apparatus', 'Lysosome', 'Vacuole']

		plt.xticks(tick_marks, classes, rotation=60)
		plt.yticks(tick_marks, classes)

		thresh = confusion_mat.max() / 2.
		for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
			plt.text(j, i, confusion_mat[i, j], horizontalalignment="center",
					 color="white" if confusion_mat[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True location')
		plt.xlabel('Predicted location')
		plt.show()
