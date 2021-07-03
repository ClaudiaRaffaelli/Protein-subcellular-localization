from collections import Counter
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Hyperparameters
n = 3
stride = 3


def createVocabulary():
    dataset_swiss = "data/swissProt.fasta"
    dataset_deeploc = "data/deeploc_data.fasta"

    with open(dataset_swiss, "r") as f:
      lines = f.readlines()
      sequences_swiss = [seq.replace("\n","") for i,seq in enumerate(lines) if i%2!=0]
    with open(dataset_deeploc, "r") as f:
      lines = f.readlines()
      sequences_deeploc = [seq.replace("\n","") for i,seq in enumerate(lines) if i%2!=0]
    sequences = sequences_swiss + sequences_deeploc

    slice_size = 1000   # evaluate slice_size sequence at a time
    token_counter = Counter()   # Count tokens' occurrences

    for batch in range(0, len(sequences), slice_size):
        tokens = []
        for seq in sequences[batch:batch + slice_size]:
            x = [seq[i:i + n] for i in range(0, len(seq), stride)]
            tokens.append(x)
        tokens = np.concatenate(tokens, axis=0)
        tokens = [x for x in tokens if len(x) == n]
        token_counter += Counter(tokens)

    with open("data/vocabulary_" + str(n) + "_" + str(stride) + ".txt", "w+") as f:
        tokens_name = np.array(list(token_counter.most_common()))[:, 0]
        tokens_name = np.insert(tokens_name, 0, ['<S>', '</S>', '<UNK>'])   # insert these mandatory tokens
        for token in tokens_name:
            f.write(token + "\n")


def createTrainingDataset():
    dataset = "data/swissProt.fasta"

    with open(dataset, "r") as f:
        lines = f.readlines()
        sequences = [seq.replace("\n", "") for i, seq in enumerate(lines) if i % 2 != 0]

    slice_size = 100
    training_slice = int(((len(sequences)*80)/100)/slice_size)  # 80%
    print("Training slices: " + str(training_slice))

    for i, batch in enumerate(range(0, int(len(sequences)), slice_size)):
        if i < training_slice:  # training set
            if not os.path.exists("data/training_" + str(n) + "_" + str(stride) + "/"):
                os.makedirs("data/training_" + str(n) + "_" + str(stride) + "/")
            with open("data/training_" + str(n) + "_" + str(stride) + "/" + str(i) + ".txt", "w+") as f:
                for i, seq in enumerate(sequences[batch:batch + slice_size]):
                    x = [seq[i:i + n] for i in range(0, len(seq), stride)]
                    for n_gram in x:
                        if len(n_gram) == n:
                            f.write(n_gram + " ")
                    if i != slice_size - 1:
                        f.write("\n")
        else:   # heldout set
            if not os.path.exists("data/heldout_" + str(n) + "_" + str(stride) + "/"):
                os.makedirs("data/heldout_" + str(n) + "_" + str(stride) + "/")
            with open("data/heldout_" + str(n) + "_" + str(stride) + "/" + str(i) + ".txt", "w+") as f:
                for i, seq in enumerate(sequences[batch:batch + slice_size]):
                    x = [seq[i:i + n] for i in range(0, len(seq), stride)]
                    for n_gram in x:
                        if len(n_gram) == n:
                            f.write(n_gram + " ")
                    if i != slice_size - 1:
                        f.write("\n")


createVocabulary()
createTrainingDataset()
