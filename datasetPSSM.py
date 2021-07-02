import numpy as np
import math
import glob
from random import sample

sequence_len = int(input("Choose between length 1000 or 400: "))
while (sequence_len != 400) and (sequence_len != 1000):
    sequence_len = int(input("Choose between length 1000 or 400: "))
print("Your choice of sequence length: {}".format(sequence_len))


labels_dic_location = {
    'Cell-membrane': 0,
    'Cytoplasm': 1,
    'Endoplasmic-reticulum': 2,
    'Golgi-apparatus': 3,
    'Lysosome_Vacuole': 4,
    'Mitochondrion': 5,
    'Nucleus': 6,
    'Peroxisome': 7,
    'Plastid': 8,
    'Extracellular': 9
}

labels_dic_membrane = {
    'M': 0,
    'S': 1,
    'U': 2
}

# Amino acid alphabet order: A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V

if sequence_len == 400:
    y_val_location = []
    y_val_membrane = []
    y_train_location = []
    y_train_membrane = []
    X_val = []
    X_train = []
    out_train = "dataset/psi-blast/results/" + str(sequence_len) + "_train"
    out_val = "dataset/psi-blast/results/" + str(sequence_len) + "_val"
else:
    y_train_val_location = []
    y_train_val_membrane = []
    X_train_val = []  # of the form (number of sequences x sequence_len x 20) where 20 is the number of amino acids
    partition = []
    out_train_val = "dataset/psi-blast/results/" + str(sequence_len) + "_train_val"

X_test = []  # of the form (number of sequences x sequence_len x 20) where 20 is the number of amino acids
y_test_location = []
y_test_membrane = []

out_test = "dataset/psi-blast/results/" + str(sequence_len) + "_test"

# needed to create k-fold validation on train_val with k=4
part = 1
# in this case train and val are saved separated and are considered less sequences
if sequence_len == 400:
    pssm_files = sample(glob.glob("dataset/DeepLoc/psi-blast/pssm/*"), 3800)
else:
    pssm_files = glob.glob("dataset/DeepLoc/psi-blast/pssm/*")

for pssm_file in pssm_files:

    # the encoded amino acid sequence will be of size sequence_len (either 400 or 1000 in our case)
    encoded_sequence = []
    name = pssm_file.split("/", 4)[3]

    # skipping the class of the dataset Cytoplasm-Nucleus since it is not relevant
    if name.split("-", 2)[2].startswith("Cytoplasm-Nucleus") or name.split("-", 3)[3].startswith("Cytoplasm-Nucleus"):
        continue

    loc = name.split("-")
    if loc[2] in labels_dic_location.keys():
        location = labels_dic_location[loc[2]]
        membrane = labels_dic_membrane[loc[3][0]]  # M for membrane, U for unknown, S for soluble
    elif str(loc[2]+"-"+loc[3]) in labels_dic_location.keys():
        location = labels_dic_location[loc[2]+"-"+loc[3]]
        membrane = labels_dic_membrane[loc[4][0]]  # M for membrane, U for unknown, S for soluble
    elif str(loc[3]) in labels_dic_location.keys():
        location = labels_dic_location[loc[3]]
        membrane = labels_dic_membrane[loc[4][0]]  # M for membrane, U for unknown, S for soluble
    else:
        location = labels_dic_location[loc[3]+"-"+loc[4]]
        membrane = labels_dic_membrane[loc[5][0]]  # M for membrane, U for unknown, S for soluble
    # print("location: {}, membrane: {}".format(location, membrane))

    # parsing the whole pssm file of psi-blast
    with open(pssm_file) as f:
        # skipping the first three lines and last six that are not part of the matrix
        lines = f.readlines()[3:-6]

        extra = len(lines) - sequence_len
        # if the sequence is longer that the max allowed we remove amino acids from the center of the sequence,
        if extra >= 0:
            index_i = math.floor(len(lines) / 2) - math.floor(extra / 2)
            index_f = math.floor(len(lines) / 2) + math.ceil(extra / 2)
            extra = 0
        # if the sequence is shorter than sequence_len is added padding at the end of the sequence
        else:
            index_i = index_f = math.floor(len(lines) / 2)
            extra = -extra

        for i in range(0, index_i):
            splitted = lines[i].split()

            # taking only the values in the matrix and doing normalization between 0 and 1
            # splitted[22:42] for right matrix and [2:22] for left matrix
            encoded_amino_acid = [float(value) / 100 for value in splitted[2:22]]
            print(splitted[2:22])
            #print(encoded_amino_acid)
            encoded_sequence.append(encoded_amino_acid)

        for i in range(index_f, len(lines)):
            splitted = lines[i].split()

            # taking only the values in the matrix and doing normalization between 0 and 1
            encoded_amino_acid = [float(value) / 100 for value in splitted[2:22]]
            #print(encoded_amino_acid)
            print(splitted[2:22])
            encoded_sequence.append(encoded_amino_acid)

        # to reach 1000 or 400 amino acid per sequence is added padding if necessary
        for i in range(extra):
            # padding of all amino acid encoded as zeros
            encoded_amino_acid = np.zeros(20)
            encoded_sequence.append(encoded_amino_acid)

        # this means that this record is for the test set.
        if pssm_file.endswith("test.pssm"):
            y_test_location.append(location)
            y_test_membrane.append(membrane)
            X_test.append(encoded_sequence)
        else:  # ==2 and it is either for training or validation
            if sequence_len == 400:
                # if modulus of part is 1 we insert the sequence in validation, otherwise is for training
                if ((((part - 1) % 4 + 4) % 4) + 1) == 1:
                    y_val_location.append(location)
                    y_val_membrane.append(membrane)
                    X_val.append(encoded_sequence)
                else:
                    y_train_location.append(location)
                    y_train_membrane.append(membrane)
                    X_train.append(encoded_sequence)
            else:
                y_train_val_location.append(location)
                y_train_val_membrane.append(membrane)
                X_train_val.append(encoded_sequence)
                partition.append((((part - 1) % 4 + 4) % 4) + 1)
            part += 1

if sequence_len == 400:
    np.savez_compressed(out_train, X_train=X_train, y_train_location=y_train_location, y_train_membrane=y_train_membrane)
    np.savez_compressed(out_val, X_val=X_val, y_val_location=y_val_location, y_val_membrane=y_val_membrane)
else:
    np.savez_compressed(out_train_val, X_train_val=X_train_val, y_train_val_location=y_train_val_location,
            y_train_val_membrane=y_train_val_membrane, partition=partition)

np.savez_compressed(out_test, X_test=X_test, y_test_location=y_test_location, y_test_membrane=y_test_membrane)
