folds = {0: [
    'sample_1128.bio',
    'sample_1133.bio',
    'sample_1152.bio',
    'sample_1169.bio',
    'sample_1242.bio',
    'sample_1248.bio',
    'sample_1252.bio',
    'sample_1419.bio',
    'sample_1439.bio',
    'sample_1495.bio',
    'sample_1505.bio',
    'sample_1568.bio',
    'sample_1592.bio'
],
1: 
[
    'sample_1921.bio',
    'sample_1956.bio',
    'sample_2129.bio',
    'sample_214.bio',
    'sample_2210.bio',
    'sample_2218.bio',
    'sample_223.bio',
    'sample_225.bio',
    'sample_226.bio',
    'sample_2275.bio',
    'sample_2604.bio',
    'sample_2623.bio'
],
2:
[
    'sample_2746.bio',
    'sample_2747.bio',
    'sample_2780.bio',
    'sample_2789.bio',
    'sample_2790.bio',
    'sample_2792.bio',
    'sample_343.bio',
    'sample_365.bio',
    'sample_377.bio',
    'sample_378.bio',
    'sample_380.bio',
    'sample_388.bio'
],
3:
[
    'sample_391.bio',
    'sample_392.bio',
    'sample_393.bio',
    'sample_394.bio',
    'sample_398.bio',
    'sample_402.bio',
    'sample_403.bio',
    'sample_439.bio',
    'sample_452.bio',
    'sample_476.bio',
    'sample_485.bio',
    'sample_570.bio'
],
4:
[
    'sample_579.bio',
    'sample_583.bio',
    'sample_664.bio',
    'sample_666.bio',
    'sample_687.bio',
    'sample_70.bio',
    'sample_71.bio',
    'sample_782.bio',
    'sample_930.bio',
    'sample_942.bio',
    'sample_945.bio',
    'sample_96.bio'
]}

import os
import json
import pandas as pd
bio_dir = "./data/bio_clamp/merge_clamp/"
save_dir = "./all_gatortronBase/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)

entity_list = ['familyHistory', 'cc', 'hpi.duration', 'hpi.assocSignsAndSymptoms', 
               'hpi.context', 'hpi.quality', 'hpi.severity', 'socialHistory', 
               'hpi.location', 'hpi.timing', 'pastHistory', 'hpi.modifyingFactors']

for i  in range(5):
    with open(f"save_pred_{str(i)}.json", "r") as f:
        data = json.load(f)
        preds = data["pred"]
        print(len(data["pred"]))
        total_preds = []
        for pred in preds:
            total_preds += pred
        print(len(total_preds))        
        index = 0
        total_bios = 0
        for bio_file in folds[i]:
            df = pd.read_csv(os.path.join(bio_dir, bio_file), sep="\t", header=None, quoting=3, names=["token", "tag", "clamp_tag", "clamp_tag_2"])
            df = df.drop(columns=["clamp_tag", "clamp_tag_2"])
            file_len = df.shape[0]
            pred_tags = total_preds[index: index + file_len]

            total_bios += file_len
            index += file_len
        
            if not os.path.isdir(os.path.join(save_dir, bio_file.replace(".bio", ""))):
                os.makedirs(os.path.join(save_dir, bio_file.replace(".bio", "")), exist_ok=True)

            for ent in entity_list:
                sf = df.copy()
                label_tags = list(sf["tag"])
                new_tags = [l if ent in l else "O" for l in label_tags]
                sf["tag"] = [l if ent in l else "O" for l in label_tags]
                sf["pred"] = [p if ent in p else "O" for p in pred_tags]
                sf.to_csv(os.path.join(save_dir, bio_file.replace(".bio", ""), ent), sep="\t", index=False)
        print(total_bios)