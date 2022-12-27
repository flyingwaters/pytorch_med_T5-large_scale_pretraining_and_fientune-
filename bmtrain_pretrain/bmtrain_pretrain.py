from dataset_test import pretrain_dataset
import logging
import math
import torch
import os
import sys
import time
import datasets
import bmtrain as bmt
from dataclasses import asdict, dataclass, field
from model_center.dataset import DistributedDataLoader
from tensorboardX import SummaryWriter

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import flax
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
from model_center.tokenizer import T5Tokenizer
from model_center.model import T5
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def generate_preprocess_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


# the global variables that the following functions need
model_name = "mt5-xxl"
text_column_name = "text"
train_batch_size = 1
max_seq_length = 1024 if model_name == "mt5-xxl" else 512
epochs = 30
# split the large dataset into small part to make it faster to process
preprocess_part_size = 1000

bmt.init_distributed(seed=0, zero_level=3)
model = T5.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
bmt.synchronize()


##############################################
def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_attention_mask=False)


dataset = load_dataset(
    "/raid/zyftest/project/med_T5/bmtrain_pretrain/pretrain_dataset.py", cache_dir="/raid/zyftest/cache_huggingface")
bmt.synchronize()
columns_name = dataset["train"].column_names
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=20,
    remove_columns=columns_name
)
# mean_noise_span_length = 3.0
# noise_density = 0.15
expanded_inputs_length, targets_length = compute_input_and_target_lengths(
    inputs_length=max_seq_length,
    noise_density=0.15,
    mean_noise_span_length=3.0,
)


class FlaxDataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                        noise_density: float,
                        mean_noise_span_length: float,
                        input_length: int,
                        target_length: int,
                        pad_token_id: int,
                        decoder_start_token_id: int):
        self.tokenizer = tokenizer 
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length 
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {k: np.array([examples[i][k] for i in range(len(examples))])
             for k, v in examples[0].items()}
        )
        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(
            expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(
            mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        batch["input_ids"] = self.filter_input_ids(
            input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        enc_input = []
        enc_length = []
        dec_length = []
        dec_input = []
        targets = []
        indexs = []
        for i in batch["input_ids"]:
            input_tokens = torch.zeros((self.input_length,), dtype=torch.int32)
            input_tokens[:self.input_length] = torch.tensor(i).int()
            enc_input.append(input_tokens)
            enc_length.append(torch.tensor(
                self.input_length, dtype=torch.int32))
            output = [tokenizer.pad_token_id,
                      tokenizer.convert_tokens_to_ids("<extra_id_0>")]
            length = len(output)
            output_tokens = torch.zeros(
                (self.target_length,), dtype=torch.int32)
            output_tokens[:length] = torch.tensor(output).int()
            output_length = torch.tensor(self.target_length, dtype=torch.int32)
            dec_length.append(output_length)
            dec_input.append(output_tokens)
            index = torch.zeros((self.target_length,), dtype=torch.int32)
            index[length - 1] = 1
            indexs.append(index)
        for j in batch["labels"]:
            targets.append(torch.tensor(j, dtype=torch.long))

        batch["enc_input"] = torch.stack(enc_input).cuda()
        batch["enc_length"] = torch.stack(enc_length).cuda()
        batch["dec_input"] = torch.stack(dec_input).cuda()
        batch["dec_length"] = torch.stack(dec_length).cuda()
        batch["target"] = torch.stack(targets).cuda()
        batch["index"] = torch.stack(indexs).cuda()

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - \
            np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(
            start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >=
                                   0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(
            num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(
            num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths],
                     axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * \
            expanded_inputs_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + expanded_inputs_length]
            for i in range(0, total_length, expanded_inputs_length)]
        for k, t in concatenated_examples.items()
    }
    return result


# process dataset to concate de
tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=2000,
    num_proc=20
)

data_collator = FlaxDataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3,
    input_length=max_seq_length,
    target_length=targets_length,
    pad_token_id=0,
    decoder_start_token_id=0,
)

num_train_samples = len(tokenized_datasets["train"])
# Avoid using jax.numpy here in case of TPU training
train_samples_idx = np.random.permutation(np.arange(num_train_samples))
train_batch_idx = generate_preprocess_splits(
    train_samples_idx, preprocess_part_size)


bmt.synchronize()
# get the memory usage
bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), lr=1e-3)
lr_scheduler = bmt.lr_scheduler.Noam(
    optimizer,
    start_lr=1e-5,
    warmup_iter=100,
    end_iter=-1)
loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
# before backward scale the loss up to 1024*old_loss and after backward scale the grad down to grad/1024
# in this way, to avoid the underflow during the update of optimizer

optim_manager = bmt.optim.OptimManager(loss_scale=1024)
optim_manager.add_optimizer(optimizer, lr_scheduler)
t = time.gmtime()
f_t = time.strftime("%Y-%m-%d %H:%M:%S", t)
bmt.print_rank(f"#########  training~!!! start time: {f_t}  ###########")
step_iter = 0
writer = SummaryWriter('runs/pretrain_loss') 

for epoch in range(1, epochs+1):
    for sample_num, batch_idx in enumerate(train_batch_idx):
        samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
        model_inputs = data_collator(samples)
        train_dataset = pretrain_dataset(model_inputs.data)
        train_dataloader = DistributedDataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True)
        model.train()
        for step_n, data in enumerate(train_dataloader):
            # for tensorboard 
            s_1 = time.time()
            step_iter+=1
            enc_input = data["enc_input"]
            enc_length = data["enc_length"]
            dec_input = data["dec_input"]
            dec_length = data["dec_length"]
            targets = data["target"]
            index = data["index"]
            output = model(enc_input, enc_length, dec_input,
                           dec_length, output_logits=True, return_dict=True)
            # calculate loss
            logits = output.logits
            loss = loss_func(
                logits.view(-1, logits.shape[-1]), targets.view(-1))
            # use bmt.sum_loss(loss) to gather all loss information from all distributed processes
            global_loss = bmt.sum_loss(loss).item()
            # tensorboardX gather the loss 
            # visualise the actual loss
            writer.add_scalar('loss', global_loss, global_step = step_iter)
            # like the function of pytorch why ? manager?
            # zero grad
            optim_manager.zero_grad()
            # scale loss before backward to avoid precision underflow of fp16
            optim_manager.backward(loss)
            # clip gradient norm
            grad_norm = optim_manager.clip_grad_norm(
                optimizer.param_groups, max_norm=10.0, norm_type=2)
            # step for all optimizer inside optim_manager
            optim_manager.step()
            s_2 = time.time()
            # print information only on rank 0 when distributed training
            bmt.print_rank(
                "epoch: {} | loss: {:.4f} | lr: {:.4e} | grad_norm: {:.4f} | steps: {} | time consume: {:.3f} ".format(
                    epoch,
                    global_loss,
                    lr_scheduler.current_lr,
                    grad_norm,
                    step_iter,
                    s_2-s_1,
                )
            )
            # save model and tensorboard
    if epoch==1 or (epoch>0 and epoch%10==0):
        bmt.save(model, f"transformer_mt5_xxl_ckpt-{epoch}-{step_iter}.pt")
    