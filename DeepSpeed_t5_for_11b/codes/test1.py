from simplet5 import SimpleT5
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
from pytorch_lightning.loggers import TensorBoardLogger
import json
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

weight_file_pth="/raid/zyftest/project/DeepSpeed_t5/weight"
with open(weight_file_pth, "r") as l:
    weight_list = json.load(l)

path = "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/dataset_tsv/all_train.tsv"
test_path =  "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/dataset_tsv/all_test.tsv"
val_pth = "/raid/yiptmp/nlp_prepare_dataset/med0_dataset/dataset_tsv/all_eval.tsv"
with open(path,"r") as train_reader:
    train_data = [i.strip() for i in train_reader]
with open(val_pth, "r") as val_reader:
    val_data = [k.strip() for k in val_reader]

print("train_len:",len(train_data))
print("weight len:", len(weight_list))

# valid the length
assert len(weight_list) == len(train_data), logging.info("weight length!=train_data_len~")

with open(test_path, "r") as test_reader:
    test_data = [j.strip() for j in test_reader]

torch.cuda.empty_cache()
gc.collect()
model = SimpleT5()

logger = TensorBoardLogger('../3b_logs_tensorboard', name='t5_3b_med')

model.from_pretrained(model_type="t5", model_name="/raid/yiptmp/huggingface-models/t5-3b")
model.train(train_df=train_data,
            logger=logger,
            eval_df=val_data,
            test_df=test_data, 
            source_max_token_len=512, 
            target_max_token_len=512, 
            batch_size=1, 
            max_epochs=100,
            devices=8, 
            precision=16, 
            early_stopping_patience_epochs=4,
            save_only_last_epoch=True,
            weight_file_pth=weight_file_pth,
            )
        