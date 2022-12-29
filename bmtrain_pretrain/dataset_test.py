import torch
from datasets import load_dataset
import os


class universe_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def make_input(self, tokenizer, template, max_encoder_length, max_decoder_length, label):
        input = tokenizer.encode(template)

        length = len(input)

        if length > max_encoder_length:
            input = input[-max_encoder_length:]

        input_tokens = torch.zeros((max_encoder_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.int32)

        output = [tokenizer.pad_token_id,
                  tokenizer.convert_tokens_to_ids("<extra_id_0>")]
        length = len(output)
        output_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
        output_tokens[:length] = torch.tensor(output).int()
        output_length = torch.tensor(length, dtype=torch.int32)

        label = tokenizer.encode(
            label, padding='max_length', max_length=max_decoder_length)
        target = torch.tensor(label, dtype=torch.long)

        index = torch.zeros((max_decoder_length,), dtype=torch.int32)
        index[length - 1] = 1

        self.data.append({
            "enc_input": input_tokens.cuda(),
            "enc_length": input_length.cuda(),
            "dec_input": output_tokens.cuda(),
            "dec_length": output_length.cuda(),
            "targets": target.cuda(),
            "index": index.cuda(),
        })

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test':
            return
        if split == 'dev':
            split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines):
                yield json.loads(row)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class truthful(universe_dataset):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()
        dataset = load_dataset(path, split=split)
        for i in dataset:
            template = i['text']
            label = i["truthful"]
            self.make_input(tokenizer, template,
                            max_encoder_length, max_decoder_length, label)


class pretrain_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        #
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {"input_ids": self.data["input_ids"][idx],
                "enc_input": self.data["enc_input"][idx],
                "dec_input": self.data["dec_input"][idx],
                "enc_length": self.data["enc_length"][idx],
                "dec_length": self.data["dec_length"][idx],
                "target": self.data["target"][idx],
                "index": self.data["index"][idx]
                }
