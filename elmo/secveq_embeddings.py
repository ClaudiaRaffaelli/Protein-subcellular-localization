from collections import Counter
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf

from IPython.display import HTML, display
import time

from bilmtf.bilm import Batcher, BidirectionalLanguageModel, weight_layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import scipy.spatial.distance as ds
import json

# Hyperparameters (I've tried n=5 and colab crashed)
n = 3
stride = 3

# n_characters must be changed from 261 to 262 before prediction
options_path = "./data/secveq/options.json"
with open(options_path) as f:
    options = json.loads(f.read())
options['char_cnn']['n_characters'] = 262

with open(options_path, 'w') as json_file:
    json.dump(options, json_file)

# Location of pretrained LM.  Here we use the test fixtures.
vocab_file = "./data/vocabulary_" + str(n) + "_" + str(stride) + ".txt"
options_file = "./data/secveq/options.json"
weight_file = "./data/secveq/weights.hdf5"

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    # Get ops to compute the LM embeddings.
    context_embeddings_op = bilm(context_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

deeploc_file = "./data/deeploc_data.fasta"

labels_dic_location = {
    'Cell.membrane': 0,
    'Cytoplasm': 1,
    'Endoplasmic.reticulum': 2,
    'Golgi.apparatus': 3,
    'Lysosome/Vacuole': 4,
    'Mitochondrion': 5,
    'Nucleus': 6,
    'Peroxisome': 7,
    'Plastid': 8,
    'Extracellular': 9
}

# Now we can compute embeddings.
with open(deeploc_file, "r") as f:
    lines = f.readlines()
    sequences_deeploc = [seq.replace("\n", "") for i, seq in enumerate(lines) if i % 2 != 0]
    headers_deeploc = [seq.replace("\n", "") for i, seq in enumerate(lines) if i % 2 == 0]

tokenized_context = []
for seq in sequences_deeploc:
    x = [seq[i:i + n] for i in range(0, len(seq), stride)]
    tokenized_context.append(x)
print(len(tokenized_context))

embeddings_path = "./data/embeddings_secveq_" + str(n) + "_" + str(stride) + "/"
slice_size = 1  # Don't change this value

# restart from where you left
# create embeddings folder if it doesn't exist
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
if os.path.isfile(embeddings_path + "sequence_completed_" + str(n) + "_" + str(stride) + ".txt"):
    with open(embeddings_path + "sequence_completed_" + str(n) + "_" + str(stride) + ".txt", "r") as f:
        sequence_completed = int(f.readline())
    X_train = np.load(embeddings_path + "X_train_" + str(n) + "_" + str(stride) + ".npy")
    y_train_subcellular = np.load(embeddings_path + "y_train_subcellular_" + str(n) + "_" + str(stride) + ".npy")
    y_train_membrane = np.load(embeddings_path + "y_train_membrane_" + str(n) + "_" + str(stride) + ".npy")
    X_test = np.load(embeddings_path + "X_test_" + str(n) + "_" + str(stride) + ".npy")
    y_test_subcellular = np.load(embeddings_path + "y_test_subcellular_" + str(n) + "_" + str(stride) + ".npy")
    y_test_membrane = np.load(embeddings_path + "y_test_membrane_" + str(n) + "_" + str(stride) + ".npy")
else:
    sequence_completed = 0
    X_train = np.zeros((1, 1024))
    y_train_subcellular = np.ones((1)) * 99
    y_train_membrane = np.ones((1))
    X_test = np.zeros((1, 1024))
    y_test_subcellular = np.ones((1)) * 99
    y_test_membrane = np.ones((1)) * 99

# start extracting
with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())
    for i, batch in enumerate(range(sequence_completed, len(tokenized_context), slice_size)):
        tokens = tokenized_context[batch:batch + slice_size]
        header_tokens = headers_deeploc[batch:batch + slice_size][0]
        print(header_tokens)
        headers_class = [val for key, val in labels_dic_location.items() if key in header_tokens]
        if "-U" not in header_tokens and len(headers_class) == 1:
            # Create batches of data.
            context_ids = batcher.batch_sentences(tokens)
            # print("Shape of context ids = ", context_ids.shape)

            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_context_input_ = sess.run(
                elmo_context_input['weighted_op'],
                feed_dict={context_character_ids: context_ids}
            )
            elmo_context_input_ = sess.run(tf.reduce_mean(elmo_context_input_, 1))
            print("Shape of generated embeddings = ", elmo_context_input_.shape)

            # Save
            if "test" not in header_tokens:
                X_train = np.concatenate((X_train, elmo_context_input_), axis=0)
                y_train_subcellular = np.concatenate((y_train_subcellular, headers_class), axis=0)
                np.save(embeddings_path + "X_train_" + str(n) + "_" + str(stride) + ".npy", X_train)
                np.save(embeddings_path + "y_train_subcellular_" + str(n) + "_" + str(stride) + ".npy",
                        y_train_subcellular)
                if "-M" in header_tokens:
                    y_train_membrane = np.concatenate((y_train_membrane, [0]), axis=0)
                else:
                    y_train_membrane = np.concatenate((y_train_membrane, [1]), axis=0)
                np.save(embeddings_path + "y_train_membrane_" + str(n) + "_" + str(stride) + ".npy", y_train_membrane)
            else:
                X_test = np.concatenate((X_test, elmo_context_input_), axis=0)
                y_test_subcellular = np.concatenate((y_test_subcellular, headers_class), axis=0)
                np.save(embeddings_path + "X_test_" + str(n) + "_" + str(stride) + ".npy", X_test)
                np.save(embeddings_path + "y_test_subcellular_" + str(n) + "_" + str(stride) + ".npy",
                        y_test_subcellular)
                if "-M" in header_tokens:
                    y_test_membrane = np.concatenate((y_test_membrane, [0]), axis=0)
                else:
                    y_test_membrane = np.concatenate((y_test_membrane, [1]), axis=0)
                np.save(embeddings_path + "y_test_membrane_" + str(n) + "_" + str(stride) + ".npy", y_test_membrane)

        sequence_completed += slice_size
        with open(embeddings_path + "sequence_completed_" + str(n) + "_" + str(stride) + ".txt", "w+") as f:
            f.write(str(sequence_completed))

        print("Sequence completed {}/{}".format(sequence_completed,len(tokenized_context)))


X_train = np.load(embeddings_path+"X_train_"+str(n)+"_"+str(stride)+".npy")
y_train_subcellular = np.load(embeddings_path+"y_train_subcellular_"+str(n)+"_"+str(stride)+".npy")
y_train_membrane = np.load(embeddings_path+"y_train_membrane_"+str(n)+"_"+str(stride)+".npy")
X_test = np.load(embeddings_path+"X_test_"+str(n)+"_"+str(stride)+".npy")
y_test_subcellular = np.load(embeddings_path+"y_test_subcellular_"+str(n)+"_"+str(stride)+".npy")
y_test_membrane = np.load(embeddings_path+"y_test_membrane_"+str(n)+"_"+str(stride)+".npy")

np.savez_compressed(embeddings_path+"train_"+str(n)+"_"+str(stride), X_train=X_train, y_train_location=y_train_subcellular, y_train_membrane=y_train_membrane)
np.savez_compressed(embeddings_path+"test_"+str(n)+"_"+str(stride), X_test=X_test, y_test_location=y_test_subcellular, y_test_membrane=y_test_membrane)
