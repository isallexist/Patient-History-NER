import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert")
parser.add_argument("--sample_id", type=str, default="all")
args = parser.parse_args()

label_dir = '/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/merge_clamp'
output_dir = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/bert_clamp_results"
sample_list = os.listdir(output_dir)
# print(sample_list)

# default = All, calculate error for all sample
# if specific sample_id is given, calculate error only for this sample, ex: "sample_70"
sample_id = "sample_" + args.sample_id

# entity_list = ['cc', 'chronicCondition', 'familyHistory', 'hpi.assocSignsAndSymptoms', 
#                'hpi.context', 'hpi.duration', 'hpi.location', 'hpi.modifyingFactors', 
#                'hpi.quality', 'hpi.severity', 'hpi.timing', 'pastHistory', 'socialHistory']

entity_list = ['problem', 'test', 'treatment','drug','bodyloc','severity','temporal']

class Entity():
    def __init__(self, entity_type, start, end):
        self.entity_type = entity_type
        self.start = start
        self.end = end
    
    #Exact match, do not care about entity_type
    def exact_matching(self, other):
        if (isinstance(other, Entity)):
            return self.start == other.start and self.end ==  other.end
    
    #Partial match, do not care about entity_type
    def partial_matching(self, other):
        if (isinstance(other, Entity)):
            if self.start == other.start and self.end ==  other.end:
                return False
            else:
                return (self.start <= other.start and self.end > other.start) or \
                    (self.start < other.end and self.end >= other.end) or \
                    (self.start < other.start and self.end > other.end) or \
                    (self.start > other.start and self.end < other.end)
        
def extract_entity(df, entity_type):
    extracted_entities = []
    prev_tag = "O"
    start = -1
    end = 0
    num_rows = df.shape[0]
    for r_idx in range(num_rows):
        token = df.iloc[r_idx, 0]
        tag = df.iloc[r_idx, 1]
        # if entity_type == "hpi.assocSignsAndSymptoms":
        #     print(f"___________________{token}")
        if tag == "O":
            if prev_tag != "O":
                end = r_idx
                extracted_entities.append(Entity(entity_type, start, end))
                prev_tag = "O"
                # if entity_type == "hpi.assocSignsAndSymptoms":
                #     print(f"______End__1_______{token}")
                
        if tag == "B-" + entity_type:
            if prev_tag != "O":
                end = r_idx
                extracted_entities.append(Entity(entity_type, start, end))
                prev_tag = "B"
                start = r_idx
                # if entity_type == "hpi.assocSignsAndSymptoms":
                #     print(f"______End__2_______{token}")
            else:
                start = r_idx
                prev_tag = "B"
                if r_idx == num_rows - 1:
                    end = num_rows
                    extracted_entities.append(Entity(entity_type, start, end))
                    # if entity_type == "hpi.assocSignsAndSymptoms":
                    #     print(f"______End__3_______{token}")
        
        if tag == "I-" + entity_type:
                
            if r_idx == num_rows - 1:
                end = num_rows
                if start != -1:
                    extracted_entities.append(Entity(entity_type, start, end))
                # if entity_type == "hpi.assocSignsAndSymptoms":
                #     print(f"______End__4_______{token}")
    
    # if entity_type == "hpi.assocSignsAndSymptoms":
    #     for e in extracted_entities:
    #         print(df.iloc[e.start:e.end, :])
    return extracted_entities

def filter_label(tag, entity_type):
    b_tag = "B-" + entity_type
    i_tag = "I-" + entity_type
    
    if b_tag not in tag and i_tag not in tag:
        return "O"
    else:
        if b_tag in tag:
            return b_tag
        else:
            return i_tag
# Get all labels for all classes
total = []
for sample in tqdm(sample_list):
    # print(f"Sample: {sample}________________________________________")
    if sample_id != "sample_all":
        if sample != sample_id:
            continue

    label_entities = []
    label_path = os.path.join(label_dir, sample + ".bio")
    df = pd.read_csv(label_path, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    
    for entity_type in entity_list:
        if entity_type == 'temporal':
            new_df = pd.concat([df.iloc[:, 0], df.iloc[:, 3].apply(lambda x: filter_label(x, entity_type=entity_type))], axis=1)

            extracted_entities = extract_entity(new_df, entity_type)
            label_entities += extracted_entities
    total.append(len(label_entities))
# for le in label_entities:
#     print(f"Label:{le.entity_type}, {le.start}, {le.end}")
print(total)
print(sum(total))
