from collections import Counter
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from IPython.display import HTML, display
import time
import scipy.spatial.distance as ds
import json
import argparse
from bilmtf.bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilmtf.bilm.data import BidirectionalLMDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def prepareData():
    """
    Calculate the number of tokens in the training set. This value will be used in options in the main function
    :return: (int) Number of train tokens
    """
    n_train_tokens_ = 0
    training_path = "data/training_" + str(n) + "_" + str(stride) + "/"
    training_files = [f for f in listdir(training_path) if isfile(join(training_path, f))]

    for file_name in training_files:
        with open("data/training_" + str(n) + "_" + str(stride) + "/" + file_name, "r") as f:
            lines = f.readlines()
            for line_i in lines:
                n_train_tokens_ += len(line_i.split(" ")) - 1

    return n_train_tokens_


def main(args):
    """
    Start Elmo training
    :param args: array of 3 strings. 1) Path to save the checkpoint files. 2) Path to get the vocabulary.
    3) Path to get the training set
    :return:
    """

    # load the vocab
    vocab = load_vocab(args[1], 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = n_train_tokens_
    # Ignore these options if using secveq options
    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 50,
                     'n_characters': 261,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 20,
    }

    prefix = args[2]
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    tf_save_dir = args[0]
    tf_log_dir = args[0]
    # Uncomment the following lines if you want to use Uniref50 pretrained model

    # load secveq options and pretrained checkpoint
    # with open("./data/secveq/options.json") as f:
    # options_secveq = json.loads(f.read())
    # options_secveq['char_cnn']['n_characters'] = 261
    # options_secveq['n_train_tokens'] = n_train_tokens
    # with open("./data/secveq/options.json", 'w') as json_file:
    # json.dump(options_secveq, json_file)
    # checkpoint_secveq = "./data/secveq/model.ckpt-1940000"

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    # Hyperparameters
    n = 3   # length of n-grams
    stride = 3  # Stride used to create the n-grams from protein sequences

    n_train_tokens_ = prepareData()     # Get total number of training tokens

    args = ["data/checkpoint_" + str(n) + "_" + str(stride) + "/",
            "data/vocabulary_" + str(n) + "_" + str(stride) + ".txt",
            "data/training_" + str(n) + "_" + str(stride) + "/*"]

    # create checkpoint folder if it doesn't exist
    if not os.path.exists("data/checkpoint_" + str(n) + "_" + str(stride) + "/"):
        os.makedirs("data/checkpoint_" + str(n) + "_" + str(stride) + "/")

    main(args)  # Start training
