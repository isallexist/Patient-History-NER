import os
import numpy as np
import json
import torch
from torch import nn

from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader

from model import NerCLAMPBERT
from train_utils import custom_loss, eval_fold

from transformers import (
    AutoTokenizer,
    default_data_collator,
)

# For reproducible results, fix seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Define pretrainded model, load tokenizer
# PRETRAINED_MODEL = "UFNLP/gatortronS"
# PRETRAINED_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"
# PRETRAINED_MODEL = "EMBO/BioMegatron345mUncased"
# PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
# PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
PRETRAINED_MODEL = "UFNLP/gatortron-base"

# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_discharge_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_gatortronS_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_meg_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_med_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_med_large_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_clinical_20/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_bio_clinical/"
PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_gatortronBase/"
# PRETRAINED_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/saved_models_plot/all_gatortronBase_20/"

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

def tokenize_and_align_labels(examples):
    #print(examples[text_column_name])
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=max_length,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    CLAMP_labels = []
    CLAMP_labels_2 = []
    for i, label in enumerate(examples[label_column_name]):
        #print(label)
        CLAMP_label = examples[CLAMP_column_name][i]
        CLAMP_label_2 = examples[CLAMP_column_name_2][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        CLAMP_previous_word_idx = None
        CLAMP_previous_word_idx_2 = None
        label_ids = []
        CLAMP_ids = []
        CLAMP_ids_2 = []
        for word_idx in word_ids:
            #print(word_idx)
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # should use -100 for correct support number when evaluation
                #label_ids.append(label_to_id[label[word_idx]])
                label_ids.append(-100)
                #label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
            previous_word_idx = word_idx

            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                CLAMP_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != CLAMP_previous_word_idx:
                CLAMP_ids.append(CLAMP_label_to_id[CLAMP_label[word_idx]])
            else:
                # should use -100 for correct support number when evaluation
                #label_ids.append(label_to_id[label[word_idx]])
                CLAMP_ids.append(CLAMP_label_to_id[CLAMP_label[word_idx]])
                
            if word_idx is None:
                CLAMP_ids_2.append(0)
            # We set the label for the first token of each word.
            elif word_idx != CLAMP_previous_word_idx_2:
                CLAMP_ids_2.append(CLAMP_label_to_id_2[CLAMP_label_2[word_idx]])    
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # should use -100 for correct support number when evaluation
                #label_ids.append(label_to_id[label[word_idx]])
                CLAMP_ids_2.append(CLAMP_label_to_id_2[CLAMP_label_2[word_idx]])
                #label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
            CLAMP_previous_word_idx = word_idx
            CLAMP_previous_word_idx_2 = word_idx

        labels.append(label_ids)
        CLAMP_labels.append(CLAMP_ids)
        CLAMP_labels_2.append(CLAMP_ids_2)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["CLAMP_labels"] = CLAMP_labels
    tokenized_inputs["CLAMP_labels_2"] = CLAMP_labels_2
    #print(tokenized_inputs["labels"])
    return tokenized_inputs

label_all = []
pred_all = []

for fold in range(0,5):
# LOAD DATA
    print(fold)
    raw_datasets = load_dataset(f"./load_folds/fold{str(fold)}.py")
    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    #print(len(raw_datasets["validation"]))
    text_column_name = "tokens"
    CLAMP_column_name = "CLAMP_tags"
    CLAMP_column_name_2 = "CLAMP_tags_2"
    label_column_name = "ner_tags"

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}

    if isinstance(features[CLAMP_column_name].feature, ClassLabel):
        CLAMP_list = features[CLAMP_column_name].feature.names
        # No need to convert the labels since they are already ints.
        CLAMP_label_to_id = {i: i for i in range(len(CLAMP_list))}

    if isinstance(features[CLAMP_column_name_2].feature, ClassLabel):
        CLAMP_list_2 = features[CLAMP_column_name_2].feature.names
        # No need to convert the labels since they are already ints.
        CLAMP_label_to_id_2 = {i: i for i in range(len(CLAMP_list_2))}
        
    num_ner_labels = len(label_list)
    num_CLAMP_labels = len(CLAMP_list)
    num_CLAMP_labels_2 = len(CLAMP_list_2)
    #print(num_ner_labels, num_CLAMP_labels, num_CLAMP_labels_2)

    pretrain_folder = PRETRAINED_DIR + f"{str(fold)}"
    model = NerCLAMPBERT(
        num_clamp_labels=num_CLAMP_labels,
        num_clamp_labels_2=num_CLAMP_labels_2,
        num_ner_labels=num_ner_labels,
        model_name=PRETRAINED_MODEL)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(pretrain_folder, "best_model_state.bin")))
    model.to(device)
    model.eval()
    # model.resize_token_embeddings(len(tokenizer))
    # print(model)

    # Preprocessing the raw_datasets.
    # First we tokenize all the texts.
    padding = "max_length"  # if args.pad_to_max_length else False
    max_length = 256
    # Tokenize all texts and align the labels with them.

    # ['id', 'ner_tags', 'sentiment', 'tokens']
    # keep sentiment
    processed_raw_datasets = raw_datasets.map(
        # tokenize_and_align_labels, batched=True, remove_columns=raw_datasets["train"].column_names
        tokenize_and_align_labels, batched=True, remove_columns=['ner_tags', 'tokens', 'CLAMP_tags', 'CLAMP_tags_2']
    )

    eval_dataset = processed_raw_datasets["test"]
    # print(eval_dataset.column_names)
    #print(train_dataset.column_names)
    # use default for max_length padding
    data_collator = default_data_collator

    eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=data_collator,
                                batch_size=1  # args.per_device_eval_batch_size
                                )

    loss_ner_fn = custom_loss(num_ner_labels, device)

    #history = defaultdict(list)
    label_fold, pred_fold = eval_fold(
        model,
        eval_dataloader,
        tokenizer,
        loss_ner_fn,
        label_list,
        device,
        len(eval_dataset)
    )
    save_json = {}
    save_json[f"label"] = label_fold
    save_json[f"pred"] = pred_fold

    label_all += label_fold
    pred_all += pred_fold


    with open(f"./save_pred_{fold}.json", "w") as out:
        json.dump(save_json, out, indent=4)

from seqeval.metrics import classification_report
print(classification_report(label_all, pred_all, mode='strict', output_dict=False))
