import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NERDataset(Dataset):
    def __init__(self, data, tag2idx, max_seq_len, model_path):
        self.data = data
        self.tag2idx = tag2idx
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, labels = self.data[index]

        tokens, token_ids, attention_mask = self.encode_text(text)
        instance = {}
        instance["tokens"] = tokens
        instance["token_ids"] = token_ids
        instance["attention_mask"] = attention_mask

        if len(list(text)) > self.max_seq_len:
            labels = labels[:self.max_seq_len - 2]
        label_ids = [self.tag2idx[tag] for tag in labels]
        instance["label_ids"] = [0] + label_ids + [0]

        return instance

    def encode_text(self, text):
        if len(list(text)) > self.max_seq_len:
            text = text[:self.max_seq_len - 2]
        tokens = ['[CLS]'] + list(text) + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(tokens)
        return tokens, token_ids, attention_mask

    def collate_fn(self, batch):
        token_ids = [ins["token_ids"] for ins in batch]
        attention_mask = [ins["attention_mask"] for ins in batch]
        label_ids = [ins["label_ids"] for ins in batch]

        token_ids = torch.tensor(
            self.pad_ids(token_ids, self.tokenizer.pad_token_id)).long()
        attention_mask = torch.tensor(
            self.pad_ids(attention_mask, self.tokenizer.pad_token_id)).float()
        label_ids = torch.tensor(
            self.pad_ids(label_ids, self.tag2idx["O"])).long()
        return token_ids, attention_mask, label_ids

    def pad_ids(self, arrays, padding, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))
            if max_length > self.max_seq_len:
                max_length = self.max_seq_len

        arrays = [
            array + [padding] * (max_length - len(array))
            for array in arrays
        ]
        return arrays
