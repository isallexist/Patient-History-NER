import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, BertTokenizer, BertModel, AutoModelForSeq2SeqLM

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MBBERT(nn.Module):

    def __init__(self, num_ner_labels, model_name):
        super(MBBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # For NER
        self.ner_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.ner_output = nn.Linear(self.bert.config.hidden_size, num_ner_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
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

        sequence_output = embeddings[0]
        sequence_output = self.ner_dropout(sequence_output)
        logits = self.ner_output(sequence_output)
        return logits
    
    def get_embedding(
        self,
        input_ids=None,
        attention_mask=None
    ):
        embeddings = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = embeddings[0]
        #sequence_output = self.dropout(sequence_output)
        return sequence_output