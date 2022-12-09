#!/usr/bin/env python
# encoding:utf-8
# #### multitask prompts to train mixture 
import functools
import os
import time
import warnings

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
def dumping_dataset(dataset_type, dataset_name):
    def dp(split, shuffle_files="false"):
      del shuffle_files
      if split == 'train':
        ds = tf.data.TextLineDataset(
            [
            LOCAL_DATA_PTH+"/"+"{}/{}/train.tsv".format(dataset_type,dataset_name)
            ]
          )
      else:
        ds = tf.data.TextLineDataset(
            [
            LOCAL_DATA_PTH+"/"+"{}/{}/test.tsv".format(dataset_type, dataset_name)
            ]
          )
    # Split each "<t1>\t<t2>" example into (input), target) tuple.
  
      ds = ds.map(functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
      ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
      return ds
    return dp
# interface 
def data_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    return text

  def to_inputs_and_targets(ex):
    """Map {"inputs": ..., "targets": ...}->{"inputs": ner..., "targets": ...}."""
    return {
        "inputs":normalize_text(ex["input"]),
        "targets": normalize_text(ex["target"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def add_task(dataset_name, dumping_dataset):
    t5.data.TaskRegistry.remove(dataset_name)
    t5.data.TaskRegistry.add(
    dataset_name,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=dumping_dataset,
    splits=["train"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[data_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy, 
               t5.evaluation.metrics.sequence_accuracy, 
               ],
    output_features=DEFAULT_OUTPUT_FEATURES
)
# print("A few raw validation examples...")
# for ex in tfds.as_numpy(dumping_dataset("train").take(1)):
#   print(ex)

def train():
    args = parse_args() 
    file_pth = args.config_file_pth
    config = config_parser(file_pth)
    task_list_name= config.get("dataset_config","tasks_name_list_pth")
    import json
    # 载入dataset 数据条数, 用generate rate of sampling
    dataset_info_path = eval(config.get('dataset_config', 'dataset_info_pth'))
    with open(dataset_info_path, "r") as reader:
      dataset_info_dict = json.load(reader)
    try:
      with open(eval(task_list_name), "r") as f:
        categories_list =  json.load(f)
    except:
      raise Exception("wrong filename {}:dataset_config/tasks_name_list_pth".format(task_list_name))
    local_data_pth = eval(config.get("dataset_config","local_data_pth"))
    list_dataset_name = [] 
    for i in categories_list:
        task_type_pth = os.path.join(local_data_pth, i)
        tmp_datasets = os.listdir(task_type_pth)
        for j in tmp_datasets:
            list_dataset_name.append((j, dataset_info_dict[j]))
            dp_data = dumping_dataset(dataset_name=j, dataset_type=i)
            add_task(j, dumping_dataset=dp_data)
    # t5.MixtureRegistry.add define one mixture
    t5.data.MixtureRegistry.remove("multitask_all")
    
    # seqio.MixtureRegistry.add("mix1",[("task1", 1), ("task2", 7)])
    t5.data.MixtureRegistry.add(
    "multitask_all",
    list_dataset_name)
    # 

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

    FINETUNE_STEPS = eval(config.get("train_strategy", "finetune_steps"))


    model.finetune(
    mixture_or_task_name="multitask_all",
    pretrained_model_dir=MODEL_DIR,
    finetune_steps=FINETUNE_STEPS
)
# Export Model

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
  