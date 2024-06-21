from transformers import BertModel, AutoModelForTokenClassification, BertTokenizer, AutoModel
from torch import nn
import torch
import torch.nn.functional as F

class NerCLAMPBERT(nn.Module):
    # Modify model to add CLAMP tags.
    # To avoid overlap: split CLAMP tags into 2 groups: 
    # 1: problem, test, treatment, drug
    def __init__(self, num_clamp_labels, num_clamp_labels_2, num_ner_labels, model_name):
        super(NerCLAMPBERT, self).__init__()
        self.num_clamp_labels = num_clamp_labels
        self.num_clamp_labels_2 = num_clamp_labels_2
        self.bert = AutoModel.from_pretrained(model_name)
        # For NER
        self.ner_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # 
        self.ner_output = nn.Linear(self.bert.config.hidden_size +  self.num_clamp_labels + self.num_clamp_labels_2 - 1, num_ner_labels)
        torch.nn.init.normal_(self.ner_output.weight, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        CLAMP_tags=None,
        CLAMP_tags_2=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        embeddings = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        CLAMP_onehot = F.one_hot(CLAMP_tags, num_classes=self.num_clamp_labels).float()
        CLAMP_onehot_2 = F.one_hot(CLAMP_tags_2, num_classes=self.num_clamp_labels_2).float()
        # print(CLAMP_tags_2.shape)
        # print(CLAMP_onehot.shape)
        # print(CLAMP_onehot_2.shape)

        # CLAMP_embedding = self.CLAMP_embedding(CLAMP_onehot)
        sequence_output = embeddings[0]
        #sequence_output += CLAMP_embedding
        sequence_output = torch.cat((sequence_output, CLAMP_onehot, CLAMP_onehot_2[:, :, 1:]), 2)
        # print(sequence_output.shape)
        #sequence_output = torch.cat((sequence_output, CLAMP_embedding), 2)
        sequence_output = self.ner_dropout(sequence_output)
        logits = self.ner_output(sequence_output)
        return logits


