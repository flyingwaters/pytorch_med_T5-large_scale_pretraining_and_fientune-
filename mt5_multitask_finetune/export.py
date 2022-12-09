#!/usr/bin/env python
# encoding:utf-8
# #### multitask prompts to train mixture 
import functools
import os
import time
import warnings
from multilingual_t5 import vocab

import seqio
import importlib
importlib.import_module("multilingual_t5.tasks")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from multilingual_t5.tasks import *
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import t5
import gin

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="config_file_name")
    parser.add_argument('--config_file_pth', type=str, default='', required=True, help='the strageties and hyperparams of training, dataset, model ')
    args = parser.parse_args()
    return args

def config_parser(path):
    import configparser
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config

@contextmanager
def tf_verbosity_level(level):
  # logging setting   
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

args = parse_args() 
file_pth = args.config_file_pth
config = config_parser(file_pth)
LOCAL_DATA_PTH = eval(config.get("dataset_config","local_data_pth"))
local_model_pth = eval(config.get("model_dir", "path"))
tf.disable_v2_behavior()      
gin.parse_config_file(local_model_pth+'/'+'operative_config.gin')

def train():
    args = parse_args() 
    file_pth = args.config_file_pth
    config = config_parser(file_pth)


    MODEL_DIR = local_model_pth
    tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
    model = t5.models.MtfModel(
  tpu=None,
  model_dir = MODEL_DIR,
  mesh_shape = eval(config.get("train_strategy","mesh_shape")),
  mesh_devices = eval(config.get("train_strategy", "mesh_devices")),
  batch_size = eval(config.get("train_strategy", "batch_size")),
  sequence_length = eval(config.get("train_strategy", "sequence_length")),
  learning_rate_schedule = eval(config.get("train_strategy", "learning_rate_schedule")),
  save_checkpoints_steps = eval(config.get("train_strategy", "save_checkpoints_steps")),
  keep_checkpoint_max = eval(config.get("train_strategy", "keep_checkpoint_max")),
  iterations_per_loop = eval(config.get("train_strategy", "iterations_per_loop")),
  
)

#     FINETUNE_STEPS = eval(config.get("train_strategy", "finetune_steps"))

#     model.finetune(
#     mixture_or_task_name="multitask_all",
#     pretrained_model_dir=MODEL_DIR,
#     finetune_steps=FINETUNE_STEPS
# )
# # Export Model

    export_dir = eval(config.get("export_strategy", "export_dir"))
    model.batch_size = eval(config.get("export_strategy", "batch_size")) # make one prediction per call
    
    saved_model_path = model.export(
    export_dir,
    checkpoint_step= eval(config.get("export_strategy", "checkpoint_step")),  # use most recent
    beam_size= eval(config.get("export_strategy", "beam_size")),  # no beam search
    # use beam search to generate more prediction 
    # which is convinient to filtering 
    temperature=eval(config.get("export_strategy", "temperature")), 
    # sample according to predicted distribution
    )
    print("Model saved to:", saved_model_path)


if __name__ == "__main__":
  train()
  