import os
import pandas as pd

T1_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/1st_tag/"
T2_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/2nd_tag/"
SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/merge_clamp/"

list_files = os.listdir(T1_DIR)

for fi in list_files:
    f = open(os.path.join(SAVE_DIR, fi), "w")
    T1_df = pd.read_csv(os.path.join(T1_DIR, fi), sep="\t", names=["token", "tag", "tag1"])
    T2_df = pd.read_csv(os.path.join(T2_DIR, fi), sep="\t", names=["token", "tag", "tag2"])
    f1 = open(os.path.join(T1_DIR, fi), "r")
    f1_lines = f1.readlines()
    f2 = open(os.path.join(T2_DIR, fi), "r")
    f2_lines = f2.readlines()
    # print(T1_df)
    for idx, line in enumerate(f1_lines):
        if line != "\n":
            new_line = line.rstrip() + "\t" + f2_lines[idx].split("\t")[-1].rstrip()
        else:
            new_line = "\n"
        f.write(new_line + "\n")
        
    # for i in range(T1_df.shape[0]):
    #     token = T1_df.loc[i, "token"]
    #     tag = T1_df.loc[i, "tag"]
    #     tag1 = T1_df.loc[i, "tag1"]
    #     tag2 = T1_df.loc[i, "tag2"]
    #     f.write("\t".join([token, tag, tag1, tag2]) + "\n")


from sklearn.model_selection import KFold

DATA_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/merge_clamp/"
SAVE_DIR = "/home/hieunghiem/CHSI/MB_BERT_new/CLAMP/data/bio_clamp/test_folds/"
NUM_FOLD = 5

if not os.path.isdir(os.path.join(SAVE_DIR, "folds")):
    os.mkdir(os.path.join(SAVE_DIR, "folds"))

for i in range(NUM_FOLD):
    if not os.path.isdir(os.path.join(SAVE_DIR, "folds", str(i))):
        os.mkdir(os.path.join(SAVE_DIR, "folds", str(i)))

data_files = sorted(os.listdir(DATA_DIR))
print(len(data_files))
print(data_files)
# data_files = ['sample_71.iob', 'sample_96.iob', 'sample_2780.iob', 'sample_1439.iob', 'sample_583.iob', 'sample_439.iob', 'sample_485.iob', 'sample_70.iob', 'sample_1248.iob', 'sample_1133.iob', 'sample_378.iob', 'sample_2218.iob', 'sample_1169.iob', 'sample_945.iob', 'sample_2275.iob', 'sample_476.iob', 'sample_2210.iob', 'sample_393.iob', 'sample_377.iob', 'sample_1956.iob', 'sample_1128.iob', 'sample_1152.iob', 'sample_403.iob', 'sample_930.iob', 'sample_2604.iob', 'sample_1495.iob', 'sample_452.iob', 'sample_388.iob', 'sample_223.iob', 'sample_2789.iob', 'sample_942.iob', 'sample_579.iob', 'sample_398.iob', 'sample_2792.iob', 'sample_402.iob', 'sample_225.iob', 'sample_1568.iob', 'sample_2746.iob', 'sample_1252.iob', 'sample_394.iob', 'sample_1419.iob', 'sample_1921.iob', 'sample_782.iob', 'sample_226.iob', 'sample_666.iob', 'sample_687.iob', 'sample_1242.iob', 'sample_391.iob', 'sample_1592.iob', 'sample_380.iob', 'sample_2623.iob', 'sample_2747.iob', 'sample_2129.iob', 'sample_1505.iob', 'sample_392.iob', 'sample_365.iob', 'sample_214.iob', 'sample_664.iob', 'sample_570.iob', 'sample_2790.iob', 'sample_343.iob']
# print(len(data_files))
kf = KFold(n_splits=NUM_FOLD)

val_fold_ids = []
# create split train_ids -> train_ids and val_ids
for fold_id, (train_ids, test_ids) in enumerate(kf.split(data_files)):
    if fold_id + 1 < NUM_FOLD:
        val_fold_ids.append(test_ids)
    else:
        val_fold_ids.insert(0,test_ids)

for fold_id, (train_ids, test_ids) in enumerate(kf.split(data_files)):
    # print(val_fold_ids[fold_id])
    # print(train_ids)
    # print(test_ids)
    print(f"Fold: {fold_id}")
    # val_ids = val_fold_ids[fold_id]
    for test_id in test_ids:
        print(data_files[test_id])
    # train_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "train.tsv")
    # train_f = open(train_path, "w")
    # test_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "test.tsv")
    # test_f = open(test_path, "w")
    # dev_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "dev.tsv")
    # dev_f = open(dev_path, "w")
    
    # for t_id in train_ids:
    #     if t_id not in val_ids:
    #         f = open(os.path.join(DATA_DIR,data_files[t_id].replace("iob", "bio")))
    #         train_f.write(f.read())
    #         f.close()
    # train_f.close()
    
    # for t_id in val_ids:
    #     f = open(os.path.join(DATA_DIR,data_files[t_id].replace("iob", "bio")))
    #     dev_f.write(f.read())
    #     f.close()
    # dev_f.close()
    
    # for t_id in test_ids:
    #     f = open(os.path.join(DATA_DIR,data_files[t_id].replace("iob", "bio")))
    #     test_f.write(f.read())
    #     f.close()
    # test_f.close()

    # # copyfile(test_path, dev_path)

