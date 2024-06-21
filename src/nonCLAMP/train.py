import numpy as np
#import wandb
import os
import datasets
import torch
from torch import nn

from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict

from model import MBBERT
from utils import train_epoch, eval_epoch, get_labels, custom_loss

import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

MODE = "all"
NUM_FOLD = 5
EPOCHS = 40
LR = 1e-5
BATCH_SIZE = 8
WEIGHT_DECAY = 0
WARMUP_STEP = 100

PADDING = "max_length"  # if args.pad_to_max_length else False
MAX_LENGTH = 256

# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_gatortronS40"
SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_bio_discharge40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_BioMegatron40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_BioMed40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_BioMed_Large40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_bio_clinical40"
# SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/saved_models_plot/" + MODE + "_gatortronBase40"

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# PRETRAINED_MODEL = "UFNLP/gatortronS"
# PRETRAINED_MODEL = "EMBO/BioMegatron345mUncased"
PRETRAINED_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"
# PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# PRETRAINED_MODEL = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
# PRETRAINED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# PRETRAINED_MODEL = "UFNLP/gatortron-base"



# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    #print(examples[text_column_name])
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=MAX_LENGTH,
        padding=PADDING,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
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

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

training_loss = np.empty((5,40))
validation_loss = np.empty((5,40))

for fold in range(NUM_FOLD):

    # if fold != 0:
    #     continue
    # run = wandb.init(reinit=True, project="mb_all_no_overlap_split311_100ep")
    # run.name = f"{MODE}_fold_{fold}"
# LOAD DATA
    # raw_datasets = load_dataset("load_folds_entity/fold_" + str(fold) + ".py")
    raw_datasets = load_dataset("/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/data/load_folds/fold_" + str(fold) + ".py")

    print(raw_datasets)
    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    # print(len(raw_datasets["validation"]))
    text_column_name = "tokens"
    label_column_name = "ner_tags"

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    print(f"A {label_to_id}")

    num_ner_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = MBBERT(
        num_ner_labels=num_ner_labels,
        model_name=PRETRAINED_MODEL)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ['id', 'ner_tags', 'sentiment', 'tokens']
    processed_raw_datasets = raw_datasets.map(
        # tokenize_and_align_labels, batched=True, remove_columns=raw_datasets["train"].column_names
        tokenize_and_align_labels, batched=True, remove_columns=['id', 'ner_tags', 'tokens']
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    # eval_dataset = processed_raw_datasets["test"]


    print(train_dataset.column_names)
    # use default for max_length padding
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=BATCH_SIZE  # args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset,
                                collate_fn=data_collator,
                                batch_size=BATCH_SIZE  # args.per_device_eval_batch_size
                                )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,  # args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)  # args.learning_rate)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEP,
        num_training_steps=total_steps
    )

    loss_ner_fn = custom_loss(num_ner_labels, device)

    #history = defaultdict(list)
    best_f1_score = 0

    for epoch in range(EPOCHS):
        print(f'Fold {fold} - Epoch {epoch + 1}/{EPOCHS}')
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
            if not os.path.isdir(os.path.join(SAVE_DIR, str(fold))):
                os.mkdir(os.path.join(SAVE_DIR, str(fold)))
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, str(fold), "best_model_state.bin"))
            best_f1_score = f1_score
    #     run.log({"Train loss": train_loss})
    #     run.log({"Train_ner_acc": ner_acc})
    #     run.log({"Eval loss": val_loss})
    #     run.log({"Eval_ner_acc": ner_val_acc})
    #     run.log({"f1_score": f1_score})
        print(f"Train loss {train_loss}")
        print(f"Train_ner_acc {ner_acc}")
        print(f"Eval loss {val_loss}")
        print(f"Eval_ner_acc {ner_val_acc}")
        print(f"f1_score {f1_score}")
        training_loss[fold][epoch] = train_loss
        validation_loss[fold][epoch] = val_loss
    # run.finish()
    print(f"Fold {fold}: Best F1_score{best_f1_score}")

np.savetxt("/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/Train_Eval_loss/training_loss_bio_discharge40.csv", training_loss, delimiter="\t")
np.savetxt("/home/hieunghiem/CHSI/MB_BERT_new/nonCLAMP/Train_Eval_loss/validation_loss_bio_discharge40.csv", validation_loss, delimiter="\t")


    