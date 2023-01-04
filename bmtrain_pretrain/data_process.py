from datasets import load_dataset
from model_center.tokenizer import T5Tokenizer
from itertools import chain


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


def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_attention_mask=False)


model_name = "mt5-large"
text_column_name = "text"
max_seq_length = 512
tokenizer = T5Tokenizer.from_pretrained(model_name)
expanded_inputs_length, targets_length = compute_input_and_target_lengths(
    inputs_length=max_seq_length,
    noise_density=0.15,
    mean_noise_span_length=3.0,
)

dataset = load_dataset(
    "/raid/zyftest/project/med_T5/bmtrain_pretrain/pretrain_dataset.py",
    cache_dir="/raid/zyftest/cache_huggingface")
columns_name = dataset["train"].column_names
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=20,
    remove_columns=columns_name
)


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


tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=2000,
    num_proc=20
)
tokenized_datasets.save_to_disk(
    "/raid/yiptmp/zh_corpus/zh_medical_tsv/processed_data_512", num_proc=10)
