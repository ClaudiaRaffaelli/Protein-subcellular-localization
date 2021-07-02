from Bio import SeqIO, SeqRecord
from Bio.Seq import Seq
import glob


labels_dic_location = {
	'plasma_membrane.fasta':'Cell.membrane',
	'cytoplasmic.fasta': 'Cytoplasm',
	'ER.fasta': 'Endoplasmic.reticulum',
	'Golgi.fasta': 'Golgi.apparatus',
	'vacuolar.fasta': 'Lysosome/Vacuole',
	'lysosomal.fasta': 'Lysosome/Vacuole',
	'mitochondrial.fasta': 'Mitochondrion',
	'nuclear.fasta': 'Nucleus',
	'peroxisomal.fasta': 'Peroxisome',
	'chloroplast.fasta': 'Plastid',
	'extracellular.fasta': 'Extracellular'
}

# we want to reserve 3 parts for train, 1 for test and 1 for validation
part = 1

sequences_list = []

# running through all fasta files in order to merge them into one
for file in glob.glob("dataset/DeepLoc/multiloc/original/*.fasta"):
	fasta_sequences = SeqIO.parse(file, 'fasta')
	for sequence in fasta_sequences:
		description, seq = sequence.description, str(sequence.seq)
		print("description: {}, sequence: {}".format(description, seq))
		# update the description in this format "id location [test]"
		split = description.split()  # something of the form "id location number number".
		new_description = split[0] + " " + labels_dic_location[file.split("/")[-1]] + "-U"
		# if the sequence is for test we annotate it
		if ((((part - 1) % 5 + 5) % 5)+1) == 1:
			new_description = new_description + " test"

		new_sequence = SeqRecord.SeqRecord(seq=Seq(seq), id=split[0], description=new_description)
		sequences_list.append(new_sequence)
		part += 1


with open("../dataset/DeepLoc/multiloc/merged_multiloc.fasta", "w") as output_handle:
	SeqIO.write(sequences_list, output_handle, "fasta")
