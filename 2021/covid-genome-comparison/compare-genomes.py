# filenames
sars_1_path = 'sars_cov_1_genome.txt'
sars_2_path = 'sars_cov_2_genome.txt'

# open each file and create a list of lines
sars_1 = open(sars_1_path, 'r').readlines()
sars_2 = open(sars_2_path, 'r').readlines()


def make_chunks(genome):
    chunk_list = []
    for i in range(len(genome)):
        chunks = genome[i].split()
        for chunk in chunks:
            chunk_list.append(chunk)
    return chunk_list


def make_string(genome):
    string = ''
    for line in genome:
        for char in line:
            if char == 'a' or char == 't' or char == 'g' or char == 'c':
                string += char
    return string


# compares the genomes directly
def direct_comparison(genome_1, genome_2):
    # new genome where all similarities are the base and all differences are dashes
    differences = []
    # number of similarities and differences between genomes 1 and 2
    same, diff = 0, 0
    # loops through indexes 0 to the length of genome 1
    for i in range(len(genome_1)):
        # ensure that line exists in both genomes
        if i < len(genome_2):
            # for each line in the genomes, create line in new differences list
            differences.append('')
            # loops through, character by character, the string at index i
            for j in range(len(genome_1[i])):
                # ensures index j is in both strings, ensures that character is not digit
                if j < len(genome_2[i]) and not genome_1[i][j].isdigit():
                    # check if character at line in each genome list is different or the same
                    if genome_1[i][j] != genome_2[i][j]:
                        differences[i] += '-'
                        # increment differences counter by 1
                        diff += 1
                    else:
                        differences[i] += genome_1[i][j]
                        # increment same counter by 1
                        same += 1

    # print each line to a separate line in terminal
    for line in differences:
        print(line, end='')

    # give the value of the function as the percentage of same characters
    return same / (diff + same)


def translate(codon):
    if codon[0] == 'a':
        if codon[1] == 'a':
            if codon[2] == 'a':
                return 'Phe'
            if codon[2] == 'g':
                return 'Phe'
            if codon[2] == 't':
                return 'Leu'
            if codon[2] == 'c':
                return 'Leu'
        if codon[1] == 'g':
            return 'Ser'
        if codon[1] == 't':
            if codon[2] == 'a':
                return 'Tyr'
            if codon[2] == 'g':
                return 'Tyr'
            if codon[2] == 't':
                return '###'
            if codon[2] == 'c':
                return '###'
        if codon[1] == 'c':
            if codon[2] == 'a':
                return 'Cys'
            if codon[2] == 'g':
                return 'Cys'
            if codon[2] == 't':
                return '###'
            if codon[2] == 'c':
                return 'Trp'
    if codon[0] == 'g':
        if codon[1] == 'a':
            if codon[2] == 'a':
                return 'Leu'
            if codon[2] == 'g':
                return 'Leu'
            if codon[2] == 't':
                return 'Leu'
            if codon[2] == 'c':
                return 'Leu'
        if codon[1] == 'g':
            return 'Pro'
        if codon[1] == 't':
            if codon[2] == 'a':
                return 'His'
            if codon[2] == 'g':
                return 'His'
            if codon[2] == 't':
                return 'Gln'
            if codon[2] == 'c':
                return 'Gln'
        if codon[1] == 'c':
            return 'Arg'

    if codon[0] == 't':
        if codon[1] == 'a':
            if codon[2] == 'a':
                return 'Ile'
            if codon[2] == 'g':
                return 'Ile'
            if codon[2] == 't':
                return 'Ile'
            if codon[2] == 'c':
                return 'Met'
        if codon[1] == 'g':
            return 'Thr'
        if codon[1] == 't':
            if codon[2] == 'a':
                return 'Asn'
            if codon[2] == 'g':
                return 'Asn'
            if codon[2] == 't':
                return 'Lys'
            if codon[2] == 'c':
                return 'Lys'
        if codon[1] == 'c':
            if codon[2] == 'a':
                return 'Ser'
            if codon[2] == 'g':
                return 'Ser'
            if codon[2] == 't':
                return 'Arg'
            if codon[2] == 'c':
                return 'Arg'
    if codon[0] == 'c':
        if codon[1] == 'a':
            return 'Val'
        if codon[1] == 'g':
            return 'Ala'
        if codon[1] == 't':
            if codon[2] == 'a':
                return 'Asp'
            if codon[2] == 'g':
                return 'Asp'
            if codon[2] == 't':
                return 'Glu'
            if codon[2] == 'c':
                return 'Glu'
        if codon[1] == 'c':
            return 'Gly'
    return '---'


def compare_amino_acids(genome_1, genome_2):
    str_1 = make_string(genome_1)
    str_2 = make_string(genome_2)

    same, diff = 0, 0

    for i in range(int(len(str_1) / 3)):
        bases_1 = str_1[i*3:(i+1)*3]
        bases_2 = str_2[i*3:(i+1)*3]
        if len(bases_1) == 3 and len(bases_2) == 3:
            codon_1 = translate(bases_1)
            codon_2 = translate(bases_2)

            if codon_1 == codon_2:
                same += 1
            else:
                diff += 1

    return same / (same + diff)


print(direct_comparison(sars_1, sars_2))

'''
sars_1_chunks = make_chunks(sars_1)
sars_2_chunks = make_chunks(sars_2)

direct_comparison(sars_1, sars_2)


def compare_chunks(chunk_list_1, chunk_list_2):
    same, diff = 0, 0
    for i in range(len(chunk_list_1)):
        if i < len(chunk_list_2):
            if not any(base.isdigit() for base in chunk_list_1[i]):
                chunk_1 = chunk_list_1[i]
                chunk_2 = chunk_list_2[i]
                for j in chunk_1:
                    if j in chunk_2:
                        chunk_2 = chunk_2[:chunk_2.index(j)] + chunk_2[chunk_2.index(j) + 1:]
                        same += 1
                        print(chunk_2)
                    else:
                        diff += 1
    return same / (diff + same)
'''


# print(compare_chunks(sars_1_chunks, sars_2_chunks))
