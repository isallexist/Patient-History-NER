from xml.dom import minidom
import os
import random as rd
import spacy
import math
nlp = spacy.load("en_core_sci_md")
from tokenizers import BertWordPieceTokenizer
bert_tokenizer = BertWordPieceTokenizer("pubmed_bert_vocab.txt")

# create a Entity class, which have type, start and end locations of entity
class Entity():
    def __init__(self, ent_type, start_ent, end_ent):
        self.ent_type = ent_type
        self.start_ent = start_ent
        self.end_ent = end_ent

# check if a sentence has entities, if Yes return location of overlapped chunk
def overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent):
    if max(start_ent, start_sent) < min(end_ent, end_sent):
        return [max(start_ent, start_sent), min(end_ent, end_sent)]
    else:
        return None

# check if a token is inside a entity, ,if Yes return location of overlapped chunk 
# same function as above, just different name for readable code
def overlap_token_in_entity(start_ent, end_ent, start_tok, end_tok):
    if max(start_ent, start_tok) < min(end_ent, end_tok):
        return [max(start_ent, start_tok), min(end_ent, end_tok)]
    else:
        return None

# extract 2 element in xmi
# element_list = ["custom:CC", "custom:Chronic"]
element_list = ["custom:Cc", "custom:Ch", "custom:Hpi", "custom:Pfsh"]
CLAMP_element_list = ["problem", "test", "treatment"]
#element_list = ["custom:ROS"]
# configs
DATA_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/xmi_data/"
CLAMP_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/clamp/"
SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/merge_data/"
FOLD_NUM = 5

# split train, validation data for each fold
total_annotation = sorted(os.listdir(DATA_DIR))
total_clamp_annotation = sorted(os.listdir(CLAMP_DIR))
# print(total_annotation)
# print(total_clamp_annotation)
assert len(total_annotation) == len(total_clamp_annotation)

test_sample_index = []
curr_index = 0
for i in range(0, FOLD_NUM):
    if i != FOLD_NUM - 1:
        curr_index += math.floor(len(total_annotation)/FOLD_NUM)
        test_sample_index.append(curr_index)
    else:
        test_sample_index.append(len(total_annotation))
print(test_sample_index)

# write data for each fold
for i in range(0, FOLD_NUM):
    fold_save_dir = os.path.join(SAVE_DIR, "fold_" + str(i))
    if not os.path.isdir(fold_save_dir):
        os.mkdir(fold_save_dir)
    # create train and test samples base on FOLD splits
    if i == 0:
        test_list = total_annotation[0: test_sample_index[0]]
        CLAMP_test_list = total_clamp_annotation[0: test_sample_index[0]]
    else:
        test_list = total_annotation[test_sample_index[i-1]: test_sample_index[i]]
        CLAMP_test_list = total_clamp_annotation[test_sample_index[i-1]: test_sample_index[i]]
    train_list = [annotation for annotation in total_annotation if annotation not in test_list]
    CLAMP_train_list =  [annotation for annotation in total_clamp_annotation if annotation not in CLAMP_test_list]

    # write train.txt, dev/test.txt
    for phase in ["train", "test"]:
        if phase == "train":
            xmi_list = train_list
            CLAMP_list = CLAMP_train_list
            save_path = os.path.join(fold_save_dir, "train.txt")
            f = open(save_path, "w")
        if phase == "test":
            xmi_list = test_list
            CLAMP_list = CLAMP_test_list
            save_path = os.path.join(fold_save_dir, "test.txt")
            dev_path = os.path.join(fold_save_dir, "dev.txt")
            f = open(save_path, "w")
            f2 = open(dev_path, "w")

        no_ent_count = 0
        ent_count = 0
        for xmi in xmi_list:
            file_path = os.path.join(DATA_DIR, xmi)
            CLAMP_path = os.path.join(CLAMP_DIR, xmi)
            print(file_path, CLAMP_path)
            tree = minidom.parse(file_path)
            CLAMP_tree = minidom.parse(CLAMP_path)

            # get text for each sample
            sample_text = tree.getElementsByTagName('cas:Sofa')[0].attributes['sofaString'].value
            
            # extract entity from XMI file
            ent_list = []
            CLAMP_ent_list = []
            for element in element_list:
                entities = tree.getElementsByTagName(element)
                for ent in entities:
                    if ent.hasAttribute("Attributes"):
                        attr = ent.attributes['Attributes'].value
                        
                        # ignore positive, negative at this time
                        if attr in ["positive", "negative"]:
                            continue
                        # ignore detail of past medical history at this time
                        if attr.startswith("pmh."):
                            continue
                        # ignore detail of chief complaint at this time
                        if attr.startswith("cc."):
                            continue
                        # ignore detail of hpi.modifyingFactors at this time
                        if attr in ["hpi.modifyingFactors.better", "hpi.modifyingFactors.noChange", 
                            "hpi.modifyingFactors.unknown", "hpi.modifyingFactors.notUseful",
                            "hpi.modifyingFactors.worse"]:
                            continue
                        if attr.startswith("ros."):
                            continue
                        
                        if attr == "chronicCondition":
                            continue
                        ent_type = attr
                        start_ent = int(ent.attributes['begin'].value)
                        end_ent = int(ent.attributes['end'].value)
                        new_ent = Entity(ent_type, start_ent, end_ent)

                        ent_list.append(Entity(ent_type, start_ent, end_ent))
                    
            for element in CLAMP_element_list:
                entities = CLAMP_tree.getElementsByTagName("typesystem:ClampNameEntityUIMA")
                for ent in entities:
                    if ent.hasAttribute("semanticTag"):
                        ent_type = ent.attributes['semanticTag'].value
                        # if ent_type == "CC_NAME" or ent_type == "Chronic_Name":
                        if ent_type == "problem" or ent_type == "test" or ent_type == "treatment":# or ent_type=="drug":
                        #if "ROS_" in ent_type:
                            start_ent = int(ent.attributes['begin'].value)
                            end_ent = int(ent.attributes['end'].value)
                            CLAMP_ent_list.append(Entity(ent_type, start_ent, end_ent))
        
            # split sample text into sentences and check entity in each sentence
            sents = list(nlp(sample_text).sents)
            for sent in sents:
                start_sent, end_sent = sent.start_char, sent.end_char

                # check a sentence have entities or not
                overlap_list = []
                for ent in ent_list:
                    start_ent, end_ent = ent.start_ent, ent.end_ent
                    overlap = overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent)
                    if overlap:
                        overlap_list.append([overlap[0], overlap[1], ent.ent_type])
                    
                CLAMP_overlap_list = []
                for ent in CLAMP_ent_list:
                    start_ent, end_ent = ent.start_ent, ent.end_ent
                    overlap = overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent)
                    if overlap:
                        CLAMP_overlap_list.append([overlap[0], overlap[1], ent.ent_type])

                if len(overlap_list) > 0:
                    ent_count += 1
                else:
                    if phase == "test":
                        no_ent_count += 1

                # split sentence into words using BertWordPieceTokenizer
                splits = bert_tokenizer.encode(sent.text)
                new_tokens, new_idxs = [], []
                for token, offset in zip(splits.tokens, splits.offsets):
                    # concatenate word pieces
                    if token.startswith("##"):
                        new_tokens[-1] = new_tokens[-1] + token[2:]
                    else:
                        new_tokens.append(token)
                        new_idxs.append(offset[0])
                new_tokens = new_tokens[1:-1]
                new_idxs = new_idxs[1:-1]
                print(len(new_tokens), len(new_idxs))
                previous_label = "O"
                CLAMP_previous_label = "O"

                for token, idx in zip(new_tokens, new_idxs):
                    if token == "\n" or token == " ":
                        continue
                    not_ent = 0
                    for overlap in overlap_list:
                        len_token = len(token)

                        if overlap_token_in_entity(overlap[0], overlap[1], start_sent + idx, start_sent + idx + len_token):
                            if previous_label == "O":
                                f.write(token + "\t" + "B-" + overlap[2] + "\t")
                                if phase == "test":
                                    f2.write(token + "\t" + "B-" + overlap[2] + "\t")
                                previous_label = "B"
                                if overlap[1] == start_sent + idx + len_token:
                                    previous_label = "O"
                            else:
                                f.write(token + "\t" + "I-" + overlap[2]+ "\t")
                                #print(token.text, token.idx, token.idx + len(token.text), "I-" + overlap[2])
                                if phase == "test":
                                    f2.write(token + "\t" + "I-" + overlap[2]+ "\t")
                                if overlap[1] == start_sent + idx + len_token:
                                    previous_label = "O"
                            break
                        else:
                            not_ent += 1
                    if not_ent == len(overlap_list):
                        f.write(token + "\t" + "O" + "\t")
                        if phase == "test":
                            f2.write(token + "\t" + "O" + "\t")
                        previous_label = "O"

                    not_ent = 0
                    for overlap in CLAMP_overlap_list:
                        len_token = len(token)

                        if overlap_token_in_entity(overlap[0], overlap[1], start_sent + idx, start_sent + idx + len_token):
                            if CLAMP_previous_label == "O":
                                f.write("B-" + overlap[2] + "\n")
                                if phase == "test":
                                    f2.write("B-" + overlap[2] + "\n")
                                CLAMP_previous_label = "B"
                                if overlap[1] == start_sent + idx + len_token:
                                    CLAMP_previous_label = "O"
                            else:
                                f.write("I-" + overlap[2]+ "\n")
                                #print(token.text, token.idx, token.idx + len(token.text), "I-" + overlap[2])
                                if phase == "test":
                                    f2.write("I-" + overlap[2]+ "\n")
                            break
                        else:
                            not_ent += 1
                    if not_ent == len(CLAMP_overlap_list):
                        f.write("O" + "\n")
                        if phase == "test":
                            f2.write("O" + "\n")
                        CLAMP_previous_label = "O"
                f.write("\n")
                if phase == "test":
                    f2.write("\n")
        f.close()

        print(no_ent_count)
        print(ent_count)