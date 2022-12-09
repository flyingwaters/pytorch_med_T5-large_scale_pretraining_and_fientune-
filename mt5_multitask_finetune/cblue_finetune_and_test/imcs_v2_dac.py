import functools
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5
import gin
tf.disable_v2_behavior()
gin.parse_config_file(
        '/raid/yiptmp/huggingface-models/cblue_finetune/mt5_3b_imcs_dac_v2/operative_config.gin'
)
# 
vocab = "/raid/yiptmp/huggingface-models/mt5_vocab/sentencepiece.model"


# Improve logging.
from contextlib import contextmanager
import logging as py_logging

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    if split == 'train':
      ds = tf.data.TextLineDataset(
            [
            '/raid/yiptmp/nlp_prepare_dataset/med0_dataset/train_cblue_v2/imcs_dac_v2/train.tsv',
            ]
          )
    else:
      ds = tf.data.TextLineDataset(
            [
            '/raid/yiptmp/nlp_prepare_dataset/med0_dataset/test_cblue_v2/imcs_dac_v2/validation.tsv',
            ]
          )
    # Split each "<t1>\t<t2>" example into (input), target) tuple.
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Map each tuple to a {"input": ... "target": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds

def ner_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    return text

  def to_inputs_and_targets(ex):
    """Map {"inputs": ..., "targets": ...}->{"inputs": ner..., "targets": ...}."""
    return {
        "inputs": normalize_text(ex["input"]),
        "targets": normalize_text(ex["target"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.remove('imcs_dac_v2')
t5.data.TaskRegistry.add(
    "imcs_dac_v2",
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=dumping_dataset,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[ner_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy, 
               t5.evaluation.metrics.sequence_accuracy, 
                ],
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab)),
    # output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab))
)

t5.data.MixtureRegistry.remove("indent_classification")
t5.data.MixtureRegistry.add(
    "indent_classification",
    ["imcs_dac_v2"],
     default_rate=1.0
)

MODEL_SIZE = "base"
BASE_PRETRAINED_DIR = "/raid/yiptmp/huggingface-models/cblue_finetune/mt5_3b_imcs_dac_v2"
MODEL_DIR = BASE_PRETRAINED_DIR

model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    mesh_shape="model:2, batch:4",
    mesh_devices=['gpu:0', 'gpu:1', 'gpu:2', 'gpu:3', 'gpu:4','gpu:5','gpu:6', 'gpu:7'],
    batch_size=16,
    sequence_length = {'inputs': 256, 'targets': 64},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=50000,
    keep_checkpoint_max=1,
    iterations_per_loop=100,#
)
FINETUNE_STEPS = 50000

model.finetune(
    mixture_or_task_name="indent_classification",
    pretrained_model_dir=MODEL_DIR,
    finetune_steps=FINETUNE_STEPS
)
export_dir = os.path.join(MODEL_DIR, "export")

model.batch_size = 1 # make one prediction per call
saved_model_path = model.export(
    export_dir,
    checkpoint_step=-1,  # use most recent
    beam_size=1,  # no beam search
    temperature=0.0,  # sample according to predicted distribution
)
print("Model saved to:", saved_model_path)