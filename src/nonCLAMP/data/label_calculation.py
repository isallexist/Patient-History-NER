#Calculate number of labels for each entity types in each note
#python label_calculation.py --sample_id 70


import os
import pandas as pd
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--sample_id", type=str)
# args = parser.parse_args()

# sample_id = 'sample_' + args.sample_id + '.iob'

def count_rows(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Use the readlines() method to get all lines in the file
        lines = file.readlines()
                # Count the number of lines
        num_rows = len(lines)
        print(num_rows)

def count_entity(file_path):
    target_values = [['B-cc', 'I-cc'],['B-familyHistory', 'I-familyHistory'],['B-hpi.assocSignsAndSymptoms', 'I-hpi.assocSignsAndSymptoms'],['B-hpi.context', 'I-hpi.context'],['B-hpi.duration', 'I-hpi.duration'],['B-hpi.location', 'I-hpi.location'],['B-hpi.modifyingFactors', 'I-hpi.modifyingFactors'],['B-hpi.quality', 'I-hpi.quality'],['B-hpi.severity', 'I-hpi.severity'],['B-hpi.timing', 'I-hpi.timing'],['B-pastHistory', 'I-pastHistory'], ['B-socialHistory', 'I-socialHistory']]
    data = pd.read_csv(file_path, sep="\t", header=None)
    data = data.drop(columns=data.columns[0])
    count_rows = []
    for entity in target_values:
        counts = len(data[data.iloc[:, 0].isin(entity)])
        count_rows.append(counts)
    print(count_rows)
    with open('output.tsv', 'a', newline='') as tsvfile:
        for item in count_rows:
            tsvfile.write(f"{item}\t")

dir_path = './bio_data/'
folder = sorted(os.listdir('./bio_data/'))

for file in folder:
    print(file)
    count_entity(os.path.join(dir_path,file))

# count_entity(os.path.join(dir_path,sample_id))



