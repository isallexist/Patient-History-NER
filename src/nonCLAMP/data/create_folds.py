import os
from shutil import copyfile
from sklearn.model_selection import KFold

DATA_DIR = "bio_data/"
SAVE_DIR = "./"
NUM_FOLD = 5

if not os.path.isdir(os.path.join(SAVE_DIR, "folds")):
    os.mkdir(os.path.join(SAVE_DIR, "folds"))

for i in range(NUM_FOLD):
    if not os.path.isdir(os.path.join(SAVE_DIR, "folds", str(i))):
        os.mkdir(os.path.join(SAVE_DIR, "folds", str(i)))

data_files = sorted(os.listdir(DATA_DIR))
print(data_files)
kf = KFold(n_splits=NUM_FOLD)

val_fold_ids = []
# create split train_ids -> train_ids and val_ids
for fold_id, (train_ids, test_ids) in enumerate(kf.split(data_files)):
    if fold_id + 1 < NUM_FOLD:
        val_fold_ids.append(test_ids)
    else:
        val_fold_ids.insert(0,test_ids)

for fold_id, (train_ids, test_ids) in enumerate(kf.split(data_files)):
    print(f"Fold {fold_id}")
    # print(val_fold_ids[fold_id])
    # print(train_ids)
    print(test_ids)
    for test_id in test_ids:
        print(data_files[test_id])
    val_ids = val_fold_ids[fold_id]
    
    train_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "train.tsv")
    train_f = open(train_path, "w")
    test_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "test.tsv")
    test_f = open(test_path, "w")
    dev_path = os.path.join(SAVE_DIR, "folds", str(fold_id), "dev.tsv")
    dev_f = open(dev_path, "w")
    
    for t_id in train_ids:
        if t_id not in val_ids:
            f = open(os.path.join(DATA_DIR,data_files[t_id]))
            train_f.write(f.read())
            f.close()
    train_f.close()
    
    for t_id in val_ids:
        f = open(os.path.join(DATA_DIR,data_files[t_id]))
        dev_f.write(f.read())
        f.close()
    dev_f.close()
    
    for t_id in test_ids:
        f = open(os.path.join(DATA_DIR,data_files[t_id]))
        test_f.write(f.read())
        f.close()
    test_f.close()

    # copyfile(test_path, dev_path)
