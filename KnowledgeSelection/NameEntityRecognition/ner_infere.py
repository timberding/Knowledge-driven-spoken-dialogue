import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ner_model import BiLSTMCRF
from ner_dataset import NERDataset
from ner_metrics import NERMetric


class NERInfere():
    def __init__(self, gpu, tag_file, pretrain_model_path, save_model_path,
                 max_seq_len):
        self.device = torch.device(gpu)

        self.tag2idx, self.idx2tag = self.load_tagid(tag_file)

        print('Load model...')
        self.model = BiLSTMCRF(hidden_dim=200, num_tags=len(self.tag2idx),
                               model_path=pretrain_model_path,
                               device=self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(save_model_path, "best_model.pt"),
                       map_location=torch.device('cpu')))
        print('Model created!')
        self.model.to(self.device)
        self.model.eval()

        self.dataset = NERDataset([], self.tag2idx, max_seq_len,
                                  pretrain_model_path)

        self.metrics = NERMetric(self.idx2tag)

    def load_tagid(self, tagfile):
        tag2idx, idx2tag = {}, {}
        with open(tagfile, 'r', encoding='utf-8') as fin:
            for line in fin:
                tag, idx = line.strip().split("\t")
                tag2idx[tag] = int(idx)
                idx2tag[int(idx)] = tag
        return tag2idx, idx2tag

    def ner(self, text):
        tokens, token_ids, attention_mask = self.dataset.encode_text(text)
        token_ids = torch.tensor(token_ids).long().to(self.device)
        attention_mask = torch.tensor(attention_mask).long().to(self.device)
        logit = self.model(token_ids=token_ids.unsqueeze(0),
                           attention_mask=attention_mask.unsqueeze(0))[0]

        ent_info = self.metrics.get_entity(logit)
        entities = []
        for info in ent_info:
            etype, start, end = info
            entity = "".join(tokens[start:end + 1])
            entities.append(entity)
        return entities
