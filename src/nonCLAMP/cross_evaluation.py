import os
import torch
import pandas as pd
from seqeval.metrics import classification_report
from sklearn.model_selection import KFold
from xml.dom import minidom

import spacy
nlp = spacy.load("en_core_sci_md")

from transformers import AutoTokenizer, default_data_collator
from tokenizers import BertWordPieceTokenizer
from model import MBBERT
from data.utils import (
    Entity, 
    Token, 
    get_actual_label_list,
    append_gold_label_dict,
    overlap_entity_in_sentence,
    overlap_token_in_entity
)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MODE = "all"
#MODE = "familyHistory"
NUM_FOLD = 5
BATCH_SIZE = 32
PADDING = "max_length"
MAX_LENGTH = 256

if MODE == "all":
    NUM_LABEL = 25#73
elif MODE == "all_non_overlap":
    NUM_LABEL = 25
else:
    NUM_LABEL = 3

# PRETRAINED_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"
# PRETRAINED_MODEL = "UFNLP/gatortronS"
# PRETRAINED_MODEL = "EMBO/BioMegatron345mUncased"
# PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
# PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# PRETRAINED_MODEL = "UFNLP/gatortron-base"


# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_gatortronS40"
# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_bio_discharge40"
# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_BioMegatron40"
# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_BioMed40"
MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_BioMed_Large40"
# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_bio_clinical40"
# MODEL_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/all_gatortronBase40"

XMI_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/xmi_data"
BIO_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/bio_data"
element_list = ["custom:Cc", "custom:Ch", "custom:Hpi", "custom:Pfsh", "custom:Ros"]

# _TRAINING_FILE = "/home/dzungle/MB_BERT/src/data/bio_data/cc_ch_hpi_pfsh/folds/0/train.tsv"
# _DEV_FILE = "/home/dzungle/MB_BERT/src/data/bio_data/cc_ch_hpi_pfsh/folds/0/dev.tsv"
# _TEST_FILE = "/home/dzungle/MB_BERT/src/data/bio_data/cc_ch_hpi_pfsh/folds/0/test.tsv"
_TRAINING_FILE = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/folds/0/train.tsv"
_DEV_FILE = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/folds/0/dev.tsv"
_TEST_FILE = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/folds/0/test.tsv"

train_df = pd.read_csv(_TRAINING_FILE, sep="\t", names=["token", "tag"])
dev_df = pd.read_csv(_DEV_FILE, sep="\t", names=["token", "tag"])
test_df = pd.read_csv(_TEST_FILE, sep="\t", names=["token", "tag"])
label_list = sorted(list(set(list(train_df["tag"].unique()) + list(dev_df["tag"].unique()) + list(test_df["tag"].unique()))))
if MODE != "all":
    if MODE != "all_non_overlap":
        label_list = ["O", "B-" + MODE, "I-" + MODE]
# get actual label list
actual_label_list = get_actual_label_list(label_list)
#print(actual_label_list)

# [true positive, actual positive, predict positive]
list_based_eval =  {k: [0, 0, 0] for k in actual_label_list}
print(NUM_LABEL)
data_files = sorted(os.listdir(BIO_DIR))
print(data_files)
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/gatortronS_results_40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/bert_results_40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/biomegatron_results_40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/biomed_results_40"
SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/Results/Models/biomed_large_results_40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/bio_clinical_results_40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/gatortronBase_results_40"



for i in range(len(data_files)):
    if not os.path.isdir(os.path.join(SAVE_DIR, data_files[i].replace(".iob", ""))):
        os.mkdir(os.path.join(SAVE_DIR, data_files[i].replace(".iob", "")))
        
kf = KFold(n_splits=NUM_FOLD)

gold_label = {k:[] for k in actual_label_list}
pred_label = {k:[] for k in actual_label_list}
for fold_id, (train_ids, test_ids) in enumerate(kf.split(data_files)):
    print(test_ids)
    # if fold_id > 0:
    #     break
    pretrain_folder = os.path.join(MODEL_DIR, str(fold_id))
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    bert_tokenizer = BertWordPieceTokenizer("/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/vocab.txt")
    model = MBBERT(num_ner_labels=len(label_list), model_name=PRETRAINED_MODEL)
    model.load_state_dict(torch.load(os.path.join(pretrain_folder, "best_model_state.bin")))
    model.to(device)
    model.eval()

    ent_statistic = {}
    for t_id in test_ids:
        print(data_files[t_id])
        pred_each_sample = {k:[] for k in actual_label_list}
        token_each_sample = []
        filepath = os.path.join(XMI_DIR,data_files[t_id].replace(".iob", ".xmi"))
        # print(filepath)
        tree = minidom.parse(filepath)
        sample_text = tree.getElementsByTagName('cas:Sofa')[0].attributes['sofaString'].value
        ent_list = []
        gold_list = {k: [] for k in actual_label_list}

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
                    
                    if MODE not in [ "all", "all_non_overlap"]:
                        if attr != MODE:
                            continue
                    ent_type = attr
                    start_ent = int(ent.attributes['begin'].value)
                    end_ent = int(ent.attributes['end'].value)
                    new_ent = Entity(ent_type, start_ent, end_ent)

                    ent_list.append(Entity(ent_type, start_ent, end_ent))
                    ent_span = sample_text[start_ent:end_ent]
                    # add to gold list of document for list based evaluation
                    if ent_span.lower() not in gold_list[ent_type]:
                        gold_list[ent_type].append(ent_span.lower())
                    if ent_type not in ent_statistic.keys():
                        ent_statistic[ent_type] = 1
                    else:
                        ent_statistic[ent_type] += 1

        # print(f'Number of entity: {len(ent_list)}')
        # print(f'gold_list: {gold_list}')
        # print("gold_list" + 50 * "_" + "\n")
        # print(gold_list)
        predict_list = {k: [] for k in actual_label_list}
        # split document into sentences using spacy
        sents = list(nlp(sample_text).sents)
        for sent in sents:
            #t_count = 0
            start_sent, end_sent = sent.start_char, sent.end_char
            # Check a sentence have entities or not
            overlap_list = []
            for ent in ent_list:
                start_ent, end_ent = ent.start_ent, ent.end_ent
                overlap = overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent)
                if overlap:
                    overlap_list.append(Token(ent.ent_type, overlap[0], overlap[1]))
            # sort overlap_list based on smaller start position and bigger position first
            overlap_list.sort(key=lambda x: (x.start_tok, -x.end_tok))

            # split sentence into words using BertWordPieceTokenizer
            splits = bert_tokenizer.encode(sent.text)
            new_tokens, new_pos = [], []
            for token, offset in zip(splits.tokens, splits.offsets):
                # concatenate word pieces
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    if token not in ["[CLS]", "[SEP]"]:
                        new_tokens.append(token)
                        new_pos.append(offset[0])


            for token, pos in zip(new_tokens, new_pos):
                # ignore newline or space token
                if token == "\n" or token == " ":
                    continue

                current_tag = "O"
                for overlap in overlap_list:
                    len_token = len(token)
                    if overlap_token_in_entity(overlap.start_tok, overlap.end_tok, start_sent + pos, start_sent + pos + len_token):
                        if current_tag == "O":
                            if overlap.start_tok == start_sent + pos:
                                current_tag = "B-" + overlap.ent_type
                            else:
                                current_tag = "I-" + overlap.ent_type
                        else:
                            if overlap.start_tok == start_sent + pos:
                                current_tag += "_B-" + overlap.ent_type
                            else:
                                current_tag += "_I-" + overlap.ent_type
                gold_label = append_gold_label_dict(gold_label, current_tag)
            
            encoding = tokenizer.encode_plus(
                sent.text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                return_token_type_ids=False,
                padding='longest',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
                )

            input_ids = encoding["input_ids"].to(device)
            outputs = model(input_ids)
            _, preds = torch.max(outputs[0], dim=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, preds):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
# To save output use this line
                    if token not in ['[PAD]', '[CLS]', '[SEP]']:
# To evaluation use this line
                    #if token != '[PAD]':
                        new_labels.append(label_list[label_idx])
                        new_tokens.append(token)
            #print(new_labels)
            for new_label in new_labels:
                pred_label = append_gold_label_dict(pred_label, new_label)
                pred_each_sample = append_gold_label_dict(pred_each_sample, new_label)
            for new_token in new_tokens:
                token_each_sample.append(new_token)
                        
            # add pred for each class
            for current_class in actual_label_list:
                span_entity = ""
                b_tag = 0
                for idx in range(0, len(new_tokens)):
                    if current_class in new_labels[idx]:
                        # some label contain more than 1 tag(class) 
                        # for example: B-chronicCondition_B-pastHistory => find tag with pastHistory
                        label_splits = new_labels[idx].split("_")
                        for split in label_splits:
                            if current_class in split:
                                tag = split
                                break
                        if "B-" in tag:
                            if span_entity != "":
                                if span_entity not in predict_list[current_class] and span_entity not in ["[CLS]", "[SEP]"]:
                                    predict_list[current_class].append(span_entity)
                                    span_entity = new_tokens[idx]
                                    b_tag = 0
                            else:
                                span_entity = new_tokens[idx]
                                b_tag = 1
                        if "I-" in tag:
                            if b_tag == 1:
                                span_entity += " " + new_tokens[idx]
                                if idx == len(new_tokens) - 1:
                                    if span_entity not in predict_list[current_class] and span_entity not in ["[CLS]", "[SEP]"]:
                                        predict_list[current_class].append(span_entity)
                    else:
                        if span_entity != "":
                            if span_entity not in predict_list[current_class] and span_entity not in ["[CLS]", "[SEP]"]:
                                    predict_list[current_class].append(span_entity)
                        span_entity = ""
                        b_tag = 0
        
        # write BIO for each entity for each file from pred_each_sample and new_tokens
        for k in actual_label_list:
            bio_f = open(os.path.join(SAVE_DIR, data_files[t_id].replace(".iob", ""), k), "w")
            for idx, token in enumerate(token_each_sample):
                bio_f.write(f'{token}\t{pred_each_sample[k][idx]}\n')
            bio_f.close()    
        print("predict_list" + 50 * "_" + "\n")
        print(predict_list)
        print("\n" *3)       

        ### List-based evaluation        
        for entity in predict_list.keys():
            predicts = predict_list[entity]
            count_pred = 0
            for pred in predicts:
                if pred in gold_list[entity]:
                    count_pred += 1
            list_based_eval[entity][0] += count_pred
            list_based_eval[entity][1] += len(gold_list[entity])
            list_based_eval[entity][2] += len(predict_list[entity])


# List based eval
print("List-based evaluation:")
total_support = 0
avg_f1 = 0
for key in list_based_eval.keys():
    print(key)
    if list_based_eval[key][2] == 0:
        precision = 0
    else:
        precision = list_based_eval[key][0]/list_based_eval[key][2]
    recall = list_based_eval[key][0]/list_based_eval[key][1]
    if precision + recall != 0:
        f1_score = 2 * precision * recall/(precision + recall)
    else:
        f1_score = 0
    print("Precision: {}, Recall: {}".format(round(precision, 2), round(recall, 2)))
    print("F1-score: {}, support: {}".format(round(f1_score, 2), list_based_eval[key][1]))
    avg_f1 += list_based_eval[key][1] * f1_score
    total_support += list_based_eval[key][1]

print(f'Average F1-score: {avg_f1/total_support}')



### BIO-based evaluation
print("BIO-based evaluation:")
total_support = 0
avg_f1 = 0
for k in gold_label.keys():
    output_dict = classification_report([gold_label[k]], [pred_label[k]], mode='strict', output_dict=True)[k]
    total_support += output_dict["support"]
    avg_f1 += output_dict["support"] * output_dict["f1-score"]
    print(f'{k}: {round(output_dict["f1-score"], 2)}')
    print(f'precision: {round(output_dict["precision"], 2)}')
    print(f'recall: {round(output_dict["recall"], 2)}')
    print(f'{round(output_dict["support"], 2)}')

print(f'Average F1-score: {avg_f1/total_support}')