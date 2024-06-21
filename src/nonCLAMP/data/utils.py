
class Entity():
    def __init__(self, ent_type, start_ent, end_ent):
        self.ent_type = ent_type
        self.start_ent = start_ent
        self.end_ent = end_ent

    def __eq__(self, other):
        if (isinstance(other, Entity)):
            return self.ent_type == other.ent_type and self.start_ent == other.start_ent and self.end_ent ==  other.end_ent

class Token():
    def __init__(self, ent_type, start_tok, end_tok):
        self.ent_type = ent_type
        self.start_tok = start_tok
        self.end_tok = end_tok

# check if a sentence has entities, return location of overlapped chunk or None
def overlap_entity_in_sentence(start_sent, end_sent, start_ent, end_ent):
    if max(start_ent, start_sent) < min(end_ent, end_sent):
        return [max(start_ent, start_sent), min(end_ent, end_sent)]
    else:
        return None

# check if a token is inside a entity, return location of overlapped chunk or None
# same function as above, just different name for readable code
def overlap_token_in_entity(start_ent, end_ent, start_tok, end_tok):
    if max(start_ent, start_tok) < min(end_ent, end_tok):
        return [max(start_ent, start_tok), min(end_ent, end_tok)]
    else:
        return None
    
def get_actual_label_list(label_list):
    actual_label_list = []
    for label in label_list:
        tags = label.split("_")
        for tag in tags:
            if tag[2:] not in actual_label_list and tag[2:] != "":
                actual_label_list.append(tag[2:])
    return actual_label_list

# gold label dict contain each class tags sequence, used to count TP, TN, FP
def append_gold_label_dict(gold_label, token_tag):
    tags = token_tag.split("_")
    for k in gold_label.keys():
        gold_label[k].append("O")
    for tag in tags:
        if tag != "O":
            gold_label[tag[2:]][-1] = tag
    # print(gold_label)
    return gold_label
    