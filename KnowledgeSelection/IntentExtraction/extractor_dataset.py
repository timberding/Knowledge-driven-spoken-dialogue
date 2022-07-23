from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DatasetExtractor(Dataset):
    def __init__(self, data, max_seq_len, model_path):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text1, text2, text3 = self.data[index]
        sample = self.tokenizer(
            [text1, text2, text3], max_length=self.max_seq_len,
            truncation=True, padding='max_length', return_tensors='pt')

        return sample

    def sequence_tokenizer(self, text):
        sample = self.tokenizer.batch_encode_plus(text,
                                                  max_length=self.max_seq_len,
                                                  truncation=True,
                                                  padding='max_length',
                                                  return_tensors='pt')

        return sample
