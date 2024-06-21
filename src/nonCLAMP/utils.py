import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from seqeval.metrics import classification_report

def get_labels(predictions, references, device):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)

    prediction_ids = [
        [p for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    label_ids = [
        [l for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    return prediction_ids, label_ids

def get_preds(predictions, ner_label_list, device):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    #print(y_pred)
    true_predictions = [ner_label_list[p] for p in y_pred[0]]

    return true_predictions


def get_true_labels(prediction_ids, label_ids, ner_label_list):
    true_predictions = [ner_label_list[p] for p in prediction_ids]
    true_labels = [ner_label_list[l] for l in label_ids]
    return true_predictions, true_labels


def custom_loss(num_ner_labels, device):
    ner_weights = num_ner_labels * [1]
    ner_weights[0] = 0.1
    loss_ner_fn = nn.CrossEntropyLoss(
        weight=torch.cuda.FloatTensor(ner_weights)).to(device)
    loss_ner_fn.ignore_index = -100
    return loss_ner_fn


def train_epoch(
    model,
    data_loader,
    tokenizer,
    loss_ner_fn,
    optimizer,
    ner_label_list,
    device,
    scheduler,
    n_examples,
):
    model = model.train()
    losses = []
    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []
    
    for d in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        ner_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print(ner_outputs.shape, targets.shape)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
            active_labels = torch.where(
                active_loss, targets.view(-1), torch.tensor(
                    loss_ner_fn.ignore_index).type_as(targets)
            )
            # print(targets.shape)
            # # torch.set_printoptions(edgeitems=10)
            # print(active_logits.shape)
            # print(active_labels.shape)
            loss = loss_ner_fn(active_logits, active_labels)
        else:
            loss =  loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)
        # predictions = ner_outputs.argmax(dim=-1)
        # print(ner_preds)
        # print(targets)
        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)


        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    #print(classification_report(label_list, pred_list, mode='strict', output_dict=True)['f1-score'])
    report = classification_report(label_list, pred_list, mode='strict', output_dict=True)
    print(report["micro avg"]["f1-score"])
    # return correct_predictions.double() / n_examples, np.mean(losses)
    return ner_correct_predictions.double()/len_ner_predict, np.mean(losses)


def eval_epoch(model,
               data_loader,
               tokenizer,
               loss_ner_fn,
               ner_label_list,
               device,
               n_examples):
    model = model.eval()
    losses = []

    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []


    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        ner_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print(ner_outputs.shape, targets.shape)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
            active_labels = torch.where(
                active_loss, targets.view(-1), torch.tensor(
                    loss_ner_fn.ignore_index).type_as(targets)
            )
            loss = loss_ner_fn(active_logits, active_labels)
        else:
            loss = loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)
        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)
        losses.append(loss.item())
    #print(n_examples)
    print(classification_report(label_list, pred_list, mode='strict'))
    report = classification_report(label_list, pred_list, mode='strict', output_dict=True)
    print(report["micro avg"]["f1-score"])
    # return correct_predictions.double() / n_examples, np.mean(losses)
    return ner_correct_predictions.double()/len_ner_predict, np.mean(losses), report["micro avg"]["f1-score"]

def eval_fold(model,
               data_loader,
               tokenizer,
               loss_ner_fn,
               ner_label_list,
               device,
               n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    ner_correct_predictions = 0
    len_ner_predict = 0
    pred_list = []
    label_list = []


    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        ner_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print(sent_output.shape, sent_target.shape)
        # print(ner_outputs.shape, targets.shape)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_outputs.view(-1, ner_outputs.shape[2])
            active_labels = torch.where(
                active_loss, targets.view(-1), torch.tensor(
                    loss_ner_fn.ignore_index).type_as(targets)
            )
            loss = loss_ner_fn(active_logits, active_labels)
        else:
            loss = loss_ner_fn(ner_outputs.view(-1, ner_outputs.shape[2]), targets.view(-1))

        ner_preds = torch.argmax(ner_outputs, dim=-1)
        ner_pred_ids, ner_label_ids = get_labels(ner_preds, targets, device)
        for (ner_pred, ner_label) in zip(ner_pred_ids, ner_label_ids):
            ner_correct_predictions += torch.sum(torch.tensor(ner_pred) == torch.tensor(ner_label))
            len_ner_predict += len(ner_pred)

            preds, labels = get_true_labels(ner_pred, ner_label, ner_label_list)
            pred_list.append(preds)
            label_list.append(labels)
        losses.append(loss.item())
    #print(n_examples)
    print(classification_report(label_list, pred_list, mode='strict'))
    report = classification_report(label_list, pred_list, mode='strict', output_dict=True)
    # return correct_predictions.double() / n_examples, np.mean(losses)
    return label_list, pred_list