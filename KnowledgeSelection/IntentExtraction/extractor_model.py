import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

torch.manual_seed(1)


class ExtractorModel(nn.Module):
    def __init__(self, device, model_path):
        super(ExtractorModel, self).__init__()
        self.device = device

        self.model = AutoModel.from_pretrained(model_path)
        for param in self.model.parameters():
            param.requires_grad = True

    def compute_loss(self, cls_emb, lamda=0.05):
        row = torch.arange(0, cls_emb.shape[0], 3, device=self.device)
        col = torch.arange(cls_emb.shape[0], device=self.device)
        col = torch.where(col % 3 != 0)[0].cuda()
        y_true = torch.arange(0, len(col), 2, device=self.device)
        similarities = F.cosine_similarity(cls_emb.unsqueeze(1),
                                           cls_emb.unsqueeze(0), dim=2)

        similarities = torch.index_select(similarities, 0, row)
        similarities = torch.index_select(similarities, 1, col)

        similarities = similarities / lamda

        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)

    def forward(self, token_ids, attention_mask,
                token_type_ids):
        outputs = self.model(token_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        cls_emb = outputs.last_hidden_state[:, 0]

        return self.compute_loss(cls_emb)

    def seq2vec(self, token_ids, attention_mask, token_type_ids):
        outputs = self.model(token_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids, return_dict=True)

        cls_emb = outputs['last_hidden_state'][:, 0]
        return cls_emb.detach().cpu().numpy()

    def text_similarity(self, data1, data2):
        seq_list_vec1 = self.seq2vec(
            token_ids=data1["input_ids"].to(self.device),
            attention_mask=data1["attention_mask"].to(self.device),
            token_type_ids=data1["token_type_ids"].to(self.device))

        seq_list_vec2 = self.seq2vec(
            token_ids=data2["input_ids"].to(self.device),
            attention_mask=data2["attention_mask"].to(self.device),
            token_type_ids=data2["token_type_ids"].to(self.device))

        similarity_list = F.cosine_similarity(seq_list_vec1, seq_list_vec2)
        score = similarity_list.detach().cpu().numpy()
        return score
