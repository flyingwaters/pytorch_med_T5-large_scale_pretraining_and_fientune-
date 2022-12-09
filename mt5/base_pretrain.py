from random import shuffle
import subprocess
import logging as py_logging
from contextlib import contextmanager
import importlib
import t5
import gin
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf
import functools
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

importlib.import_module("multilingual_t5.tasks")
# mt5 settings
tf.disable_v2_behavior()

# Improve logging.


@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)


gin.parse_config_file(
    '/raid/yiptmp/huggingface-models/mt5-base-tf-ckpt/pretrain_operative_config.gin'
)

# vocab = "gs://t5_training/models/spm/t5_bio_spm_small.model"


def dumping_dataset(split, shuffle_files=False):
    del shuffle_files
    files_zh_medical = list(map(lambda x: b'/raid/yiptmp/zh_corpus/zh_medical_tsv/'+x.strip(), subprocess.run(
        ['ls', '/raid/yiptmp/zh_corpus/zh_medical_tsv'], stdout=subprocess.PIPE).stdout.splitlines()))
    shuffle(files_zh_medical)

    print(files_zh_medical[0])

    ds = tf.data.TextLineDataset(
        files_zh_medical
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None', ex[0]])))
    ds = ds.shuffle(buffer_size=4000000)

    return ds


from multilingual_t5.tasks import *
t5.data.TaskRegistry.remove('dumping_dataset')
t5.data.TaskRegistry.add(
    'dumping_dataset',
    dataset_fn=dumping_dataset,
    splits=['train'],
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey,
        key_map={'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor=t5.data.preprocessors.unsupervised,
    # output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab)),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[],
)
t5.data.MixtureRegistry.remove('all_bioT5')
t5.data.MixtureRegistry.add(
    'all_bioT5',
    [
        'dumping_dataset',
    ],
    default_rate=1.0,
)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


MODEL_SIZE = 'base'
model_parallelism, train_batch_size, keep_checkpoint_max = {
    'small': (1, 256, 16),
    'base': (2, 128*2, 8),
    'large': (8, 64, 4),
    '3B': (8, 16, 1),
    '11B': (8, 16, 1),
}[MODEL_SIZE]


PRETRAINED_DIR = "/raid/yiptmp/huggingface-models/mt5-base-tf-ckpt"
model = t5.models.MtfModel(
    tpu=None,
    model_dir=PRETRAINED_DIR,
    mesh_shape="model:1, batch:4",
    mesh_devices=['gpu:0', 'gpu:1', 'gpu:2', 'gpu:3'],
    batch_size=16,
    sequence_length={'inputs': 512, 'targets': 512},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=100000,
    keep_checkpoint_max=1,
    iterations_per_loop=100,
)

model.train(mixture_or_task_name='all_bioT5', steps=1100000)
# export_dir = os.path.join(
#     '/raid/zyftest/project/SciFive/result_model', "mt5_large_tf")

# model.batch_size = 1  # make one prediction per call
# saved_model_path = model.export(
#     export_dir,
#     checkpoint_step=-1,  # use most recent
#     beam_size=1,  # no beam search
#     temperature=1.0,  # sample according to predicted distribution
# )
# print("Model saved to:", saved_model_path)
