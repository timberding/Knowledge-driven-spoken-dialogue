import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from extractor_model import ExtractorModel
from extractor_dataset import DatasetExtractor


class IntentInfere():
    def __init__(self, gpu, pretrain_model_path, save_model_path, max_seq_len):
        self.device = torch.device(gpu)

        print('Load model...')
        self.model = ExtractorModel(model_path=pretrain_model_path,
                                    device=self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(save_model_path, "best_model.pt"),
                       map_location=torch.device('cpu')))
        print('Model created!')
        self.model.to(self.device)
        self.model.eval()

        self.dataset = DatasetExtractor([], max_seq_len,
                                        pretrain_model_path)

    def text_smiliary(self, text1, text2):
        data1 = self.dataset.sequence_tokenizer(text1)

        seq_vec1 = self.model.seq2vec(
            token_ids=data1["input_ids"].to(self.device),
            attention_mask=data1["attention_mask"].to(
                self.device),
            token_type_ids=data1["token_type_ids"].to(
                self.device))

        data2 = self.dataset.sequence_tokenizer(
            text2)

        seq_vec2 = self.model.seq2vec(
            token_ids=data2["input_ids"].to(self.device),
            attention_mask=data2["attention_mask"].to(
                self.device),
            token_type_ids=data2["token_type_ids"].to(
                self.device))

        seq_vec1 = seq_vec1 / (seq_vec1 ** 2).sum(
            axis=1, keepdims=True) ** 0.5
        seq_vec2 = seq_vec2 / (seq_vec2 ** 2).sum(
            axis=1, keepdims=True) ** 0.5
        similarity = (seq_vec1 * seq_vec2).sum(axis=-1)
        return similarity
