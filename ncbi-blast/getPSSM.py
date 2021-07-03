import os, sys
from optparse import OptionParser

# Process command-line options
parser = OptionParser(add_help_option=False)
parser.add_option('--slice', type=str, help=('Dataset slice'))
(options, args) = parser.parse_args()

if options.slice:
	d = options.slice
else:
	d = "1"

# Choose dataset
dataset = "./query/" + d + ".fasta"

# Start psi-blast
with open(dataset) as f:
	line = f.readline()
	i = 1
	while line:
		if ">" in line:
			# header
			title = line[:-1].replace(" ", "-").replace(">", "-").replace(".", "-").replace("/", "_")
			print("Sequence " + str(i))
			print(title)
			line = f.readline()
		else:
			# Sequence			
			with open("./query/query.fasta", "w") as query:
				query.write(line[:-1])
			command = "./bin/psiblast -db ./blastdb/swissprot -query ./query/query.fasta -num_iterations 4 -out output.txt -save_pssm_after_last_round -out_ascii_pssm ./results/"+d+"_"+str(i)+"_"+title+".pssm -export_search_strategy strategy.txt"
			os.system(command)
			line = f.readline()
			i+=1
