import os
import spacy
nlp = spacy.load("en_core_sci_md")

from xml.dom import minidom
from tokenizers import BertWordPieceTokenizer
bert_tokenizer = BertWordPieceTokenizer("vocab.txt")

from utils import Entity, Token, overlap_entity_in_sentence, overlap_token_in_entity
# define list of layers(elements) you need here: 
# "custom:Sub"? ignore at this time, you can add it here
element_list = ["custom:Cc", "custom:Ch", "custom:Hpi", "custom:Pfsh", "custom:Ros"]

DATA_DIR = "xmi_data"
SAVE_DIR = "bio_data"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

xmi_list = sorted(os.listdir(DATA_DIR))

overlap_statistic = {}
ent_statistic = {}

for xmi_file in xmi_list:
    file_path = os.path.join(DATA_DIR, xmi_file)
    save_path = os.path.join(SAVE_DIR, xmi_file.replace("xmi", "iob"))
    tree = minidom.parse(file_path)
    
    # get text for each sample
    sample_text = tree.getElementsByTagName('cas:Sofa')[0].attributes['sofaString'].value

    # extract all entity from XMI file, save to a list
    ent_list = []
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
                if ent_type not in ent_statistic.keys():
                    ent_statistic[ent_type] = 1
                else:
                    ent_statistic[ent_type] += 1
                # check totally duplicate, not present in our data
                # if len(ent_list) == 0:
                #     ent_list.append(Entity(ent_type, start_ent, end_ent))
                # else:
                #     duplicate = 0
                #     for e in ent_list:
                #         if e == new_ent:
                #             print("_____________________________________________")
                #             duplicate += 1

                #     if duplicate == 0:
                #         ent_list.append(Entity(ent_type, start_ent, end_ent))

    # print(xmi_file+ "________")
    # print(ent_list[0].start_ent)
    # print(ent_list[1].start_ent)
    #       
    # count entity overlapping 
    count = 0
    print(xmi_file+ "________")
    for i in range(0, len(ent_list) - 1):
        for j in range(i, len(ent_list)):
            if i == j:
                continue
            else:
                if ent_list[i].start_ent >= ent_list[j].start_ent and ent_list[i].end_ent <= ent_list[j].end_ent:
                    print("\nOverlap detected")
                    print(ent_list[i].start_ent, ent_list[i].end_ent, ent_list[i].ent_type)
                    print(ent_list[j].start_ent, ent_list[j].end_ent, ent_list[j].ent_type)

                    if ent_list[i].ent_type > ent_list[i].ent_type:
                        overlap_type = ent_list[i].ent_type + "_" + ent_list[j].ent_type 
                    else:
                        overlap_type = ent_list[j].ent_type + "_" + ent_list[i].ent_type 
                    if overlap_type not in overlap_statistic.keys():
                        overlap_statistic[overlap_type] = 1
                    else:
                        overlap_statistic[overlap_type] += 1
                    count += 1
                if (ent_list[j].start_ent >= ent_list[i].start_ent and ent_list[j].end_ent < ent_list[i].end_ent) or (ent_list[j].start_ent > ent_list[i].start_ent and ent_list[j].end_ent <= ent_list[i].end_ent):
                    print("\nOverlap detected")
                    print(ent_list[i].start_ent, ent_list[i].end_ent, ent_list[i].ent_type)
                    print(ent_list[j].start_ent, ent_list[j].end_ent, ent_list[j].ent_type)
                    if ent_list[i].ent_type > ent_list[i].ent_type:
                        overlap_type = ent_list[i].ent_type + "_" + ent_list[j].ent_type 
                    else:
                        overlap_type = ent_list[j].ent_type + "_" + ent_list[i].ent_type 
                    if overlap_type not in overlap_statistic.keys():
                        overlap_statistic[overlap_type] = 1
                    else:
                        overlap_statistic[overlap_type] += 1
                    count += 1
    print(count)
    # break
    f = open(save_path, "w")
    # split sample text into sentences and check entity in each sentence
    sents = list(nlp(sample_text).sents)
    for sent in sents:
        start_sent, end_sent = sent.start_char, sent.end_char

        # check a sentence have entities or not, save it to a list
        overlap_list = []
        for ent in ent_list:
            start_ent, end_ent = ent.start_ent, ent.end_ent
            overlap = overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent)
            if overlap:
                overlap_list.append([overlap[0], overlap[1], ent.ent_type])
        
        # Note: I used nltk tokenizer before but it has a problem so I change to BertWordPieceTokenizer
        # split sentence into words using BertWordPieceTokenizer
        # some word will split into pieces. For example: words -> wo## and ##rds.
        # This part of code below will concatenate pieces -> original word 
        # new_token -> original word
        # new_pos -> position of token in a sentence
        splits = bert_tokenizer.encode(sent.text)
        new_tokens, new_pos = [], []
        for token, offset in zip(splits.tokens, splits.offsets):
            # concatenate word pieces
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_tokens.append(token)
                new_pos.append(offset[0])
        new_tokens = new_tokens[1:-1]
        new_pos = new_pos[1:-1]
        #print(len(new_tokens), len(new_pos))


        previous_label = "O"
        # previous_tag = "O"
        for token, pos in zip(new_tokens, new_pos):
            # ignore newline or space token
            if token == "\n" or token == " ":
                continue

            check_count = 0
            for overlap in overlap_list:
                len_token = len(token)

                if overlap_token_in_entity(overlap[0], overlap[1], start_sent + pos, start_sent + pos + len_token):
                    if previous_label == "O":
                        f.write(token + "\t" + "B-" + overlap[2] + "\n")

                        previous_label = "B"
                        # previous_tag = overlap[2]
                        if overlap[1] == start_sent + pos + len_token:
                            previous_label = "O"
                    else:
                        # if overlap[2] != previous_tag:
                        #     f.write(token + "\t" + "B-" + overlap[2]+ "\n")
                        # else:
                        f.write(token + "\t" + "I-" + overlap[2]+ "\n")
                        if overlap[1] == start_sent + pos + len_token:
                            previous_label = "O"
                        #previous_tag = overlap[2]
                    break
                else:
                    check_count += 1
            # After for loop through overlap_list, write token with tag O 
            if check_count == len(overlap_list):
                f.write(token + "\t" + "O" + "\n")
                previous_label = "O"
        
        f.write("\n")
    f.close()

print(overlap_statistic)
print(ent_statistic)

ent_overlap_dict = {} 
for ek in ent_statistic.keys():
    count = 0
    for ok in overlap_statistic.keys():
        if ek in ok:
            count += overlap_statistic[ok]
    ent_overlap_dict[ek] = count
print(ent_overlap_dict)