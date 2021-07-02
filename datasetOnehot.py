from Bio import SeqIO
import numpy as np
import math
from random import sample

sequence_len = int(input("Choose between length 1000 or 400: "))
while (sequence_len != 400) and (sequence_len != 1000):
	sequence_len = int(input("Choose between length 1000 or 400: "))
print("Your choice of sequence length: {}".format(sequence_len))

input_dataset = "dataset/DeepLoc/DeepLoc.rtf"

fasta_sequences = SeqIO.parse(open(input_dataset), 'fasta')

# in this case train and val are saved separated and are considered less sequences
if sequence_len == 400:
	fasta_sequences = sample(list(fasta_sequences), 3800)

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

labels_dic_membrane = {
	'M': 0,
	'S': 1,
	'U': 2
}

amino_acid_alphabet = {
	'A': 0,
	'C': 1,
	'D': 2,
	'E': 3,
	'F': 4,
	'G': 5,
	'H': 6,
	'I': 7,
	'K': 8,
	'L': 9,
	'M': 10,
	'N': 11,
	'P': 12,
	'Q': 13,
	'R': 14,
	'S': 15,
	'T': 16,
	'V': 17,
	'W': 18,
	'Y': 19,
}

if sequence_len == 400:
	y_val_location = []
	y_val_membrane = []
	y_train_location = []
	y_train_membrane = []
	X_val = []
	X_train = []
	out_train = "dataset/one-hot/" + str(sequence_len) + "_train"
	out_val = "dataset/one-hot/" + str(sequence_len) + "_val"
else:
	y_train_val_location = []
	y_train_val_membrane = []
	X_train_val = []  # of the form (number of sequences x sequence_len x 20) where 20 is the number of amino acids
	partition = []
	out_train_val = "dataset/one-hot/" + str(sequence_len) + "_train_val"

X_test = []  # of the form (number of sequences x sequence_len x 20) where 20 is the number of amino acids
y_test_location = []
y_test_membrane = []

out_test = "dataset/one-hot/" + str(sequence_len) + "_test"

# needed to create k-fold validation on train_val with k=4
part = 1
for fasta in fasta_sequences:
	description, sequence = fasta.description, str(fasta.seq)
	# print("description: {}, sequence: {}".format(id, description, sequence))
	split = description.split()  # something of the form "id location [test]".

	# skipping the class of the dataset Cytoplasm-Nucleus since it is not relevant
	if split[1].startswith("Cytoplasm-Nucleus"):
		continue

	location = split[1].split("-")[0]
	membrane = split[1].split("-")[1][0]  # M for membrane, U for unknown, S for soluble
	# print("location: {}, membrane: {}".format(location, membrane))

	# the encoded amino acid sequence will be of size sequence_len (either 400 or 1000 in our case)
	encoded_sequence = []
	# deleting last element of sequence since it is not part of the sequence itself (it is a placeholder /)
	sequence = sequence[:-1]

	# amino acids missing or extra to obtain a sequence of perfectly sequence_len length
	extra = len(sequence) - sequence_len
	# if the sequence is longer that the max allowed we remove amino acids from the center of the sequence,
	if extra >= 0:
		index_i = math.floor(len(sequence) / 2) - math.floor(extra / 2)
		index_f = math.floor(len(sequence) / 2) + math.ceil(extra / 2)
		extra = 0
	# if the sequence is shorter than sequence_len is added padding at the end of the sequence (all amino acids as zero
	# and none as 1)
	else:
		index_i = index_f = math.floor(len(sequence) / 2)
		extra = -extra

	# it is adopted the one-hot encoding so that each element (amino acid) of the sequence is encoded as a vector
	# with all zeros and a single 1 in correspondence of one of the 20 amino acid.
	for i in range(0, index_i):
		amino_acid = sequence[i]
		encoded_amino_acid = np.zeros(20)
		try:
			encoded_amino_acid[amino_acid_alphabet[amino_acid]] = 1
		except:
			# if the amino acid is non existing (unknown) we simply add a vector of zeros
			print("Unknown amino acid: {}, \n whole sequence: {}".format(amino_acid, sequence))
		encoded_sequence.append(encoded_amino_acid)

	for i in range(index_f, len(sequence)):
		amino_acid = sequence[i]
		encoded_amino_acid = np.zeros(20)
		try:
			encoded_amino_acid[amino_acid_alphabet[amino_acid]] = 1
		except:
			# if the amino acid is non existing (unknown) we simply add a vector of zeros
			print("Unknown amino acid: {}, \nWhole sequence: {}".format(amino_acid, sequence))
		encoded_sequence.append(encoded_amino_acid)

	# to reach 1000 or 400 amino acid per sequence is added padding if necessary
	for i in range(extra):
		# padding of all amino acid encoded as zeros
		encoded_amino_acid = np.zeros(20)
		encoded_sequence.append(encoded_amino_acid)

	# this means that this record is for the test set.
	if len(split) == 3:
		y_test_location.append(labels_dic_location[location])
		y_test_membrane.append(labels_dic_membrane[membrane])
		X_test.append(encoded_sequence)
	else:  # ==2 and it is either for training or validation
		if sequence_len == 400:
			# if modulus of part is 1 we insert the sequence in validation, otherwise is for training
			if ((((part - 1) % 4 + 4) % 4)+1) == 1:
				y_val_location.append(labels_dic_location[location])
				y_val_membrane.append(labels_dic_membrane[membrane])
				X_val.append(encoded_sequence)
			else:
				y_train_location.append(labels_dic_location[location])
				y_train_membrane.append(labels_dic_membrane[membrane])
				X_train.append(encoded_sequence)
		else:
			y_train_val_location.append(labels_dic_location[location])
			y_train_val_membrane.append(labels_dic_membrane[membrane])
			X_train_val.append(encoded_sequence)
			partition.append((((part - 1) % 4 + 4) % 4)+1)
		part += 1

if sequence_len == 400:
	np.savez_compressed(out_train, X_train=X_train, y_train_location=y_train_location, y_train_membrane=y_train_membrane)
	np.savez_compressed(out_val, X_val=X_val, y_val_location=y_val_location, y_val_membrane=y_val_membrane)
else:
	np.savez_compressed(out_train_val, X_train_val=X_train_val, y_train_val_location=y_train_val_location,
			y_train_val_membrane=y_train_val_membrane, partition=partition)

np.savez_compressed(out_test, X_test=X_test, y_test_location=y_test_location, y_test_membrane=y_test_membrane)

