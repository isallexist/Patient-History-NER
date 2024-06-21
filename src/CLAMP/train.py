import argparse
import logging
import math
import os
import random
import numpy as np
import csv

import datasets
import torch
from torch import nn

from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from model import NerCLAMPBERT
from train_utils import train_epoch, eval_epoch, get_labels, custom_loss
import argparse

import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int)
args = parser.parse_args()
# Pick a fold to train
FOLD = args.fold
EPOCHS = 40
LR = 1e-5
# PRETRAINED_MODEL = "UFNLP/gatortronS"
PRETRAINED_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"
# For reproducible results, fix seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load data using dataset.py file
raw_datasets = load_dataset(f"load_folds/fold{FOLD}.py")

if raw_datasets["train"] is not None:
    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
else:
    column_names = raw_datasets["validation"].column_names
    features = raw_datasets["validation"].features
    
text_column_name = "tokens"
CLAMP_column_name = "CLAMP_tags"
CLAMP_column_name_2 = "CLAMP_tags_2"
label_column_name = "ner_tags"

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}

if isinstance(features[CLAMP_column_name].feature, ClassLabel):
    CLAMP_list = features[CLAMP_column_name].feature.names
    CLAMP_label_to_id = {i: i for i in range(len(CLAMP_list))}

if isinstance(features[CLAMP_column_name_2].feature, ClassLabel):
    CLAMP_list_2 = features[CLAMP_column_name_2].feature.names
    CLAMP_label_to_id_2 = {i: i for i in range(len(CLAMP_list_2))}
    
num_ner_labels = len(label_list)
num_CLAMP_labels = len(CLAMP_list)
num_CLAMP_labels_2 = len(CLAMP_list_2)
print(num_ner_labels, num_CLAMP_labels, num_CLAMP_labels_2 - 1)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
model = NerCLAMPBERT(
    num_clamp_labels=num_CLAMP_labels,
    num_clamp_labels_2=num_CLAMP_labels_2,
    num_ner_labels=num_ner_labels,
    model_name=PRETRAINED_MODEL)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing the raw_datasets.
# First we tokenize all the texts.
padding = "max_length"  # if args.pad_to_max_length else False
max_length = 256
# Tokenize all texts and align the labels with them.


def tokenize_and_align_labels(examples):
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
        CLAMP_label = examples[CLAMP_column_name][i]
        CLAMP_label_2 = examples[CLAMP_column_name_2][i]
        # print(CLAMP_label_2)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        CLAMP_previous_word_idx = None
        CLAMP_previous_word_idx_2 = None
        label_ids = []
        CLAMP_ids = []
        CLAMP_ids_2 = []
        for word_idx in word_ids:
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
    return tokenized_inputs


processed_raw_datasets = raw_datasets.map(
    tokenize_and_align_labels, batched=True, remove_columns=['id', 'ner_tags', 'tokens', 'CLAMP_tags', 'CLAMP_tags_2']
)

train_dataset = processed_raw_datasets["train"]
eval_dataset = processed_raw_datasets["validation"]

# use default for max_length padding
data_collator = default_data_collator

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8  # args.per_device_train_batch_size
)
eval_dataloader = DataLoader(eval_dataset,
                             collate_fn=data_collator,
                             batch_size=8  # args.per_device_eval_batch_size
                             )

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,  # args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

loss_ner_fn = custom_loss(num_ner_labels, device)

best_f1_score = 0
training_loss = []
validation_loss = []

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    ner_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        tokenizer,
        loss_ner_fn,
        optimizer,
        label_list,
        device,
        scheduler,
        len(train_dataset)
    )
    print(f'Train loss {train_loss}, ner_acc: {ner_acc}')
    ner_val_acc, val_loss, f1_score = eval_epoch(
        model,
        eval_dataloader,
        tokenizer,
        loss_ner_fn,
        label_list,
        device,
        len(eval_dataset)
    )
    print(
        f'Eval loss {val_loss}, ner_acc: {ner_val_acc}')

    if f1_score > best_f1_score:
        # torch.save(model.state_dict(), f'./saved_models_plot/all_gatortronS/{str(FOLD)}/best_model_state.bin')
        torch.save(model.state_dict(), f'./saved_models_plot/all_bio_discharge/{str(FOLD)}/best_model_state.bin')
        best_f1_score = f1_score
    print(best_f1_score)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)

training_loss_csv = f"/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/Train_Eval_loss/training_loss_bert_{str(FOLD)}.csv"
validation_loss_csv = f"/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/Train_Eval_loss/validation_loss_bert_{str(FOLD)}.csv"

with open(training_loss_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(training_loss)

with open(validation_loss_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(validation_loss)
