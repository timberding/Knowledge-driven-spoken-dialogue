import torch
import torch.nn as nn

from transformers import AutoModel
from torchcrf import CRF

torch.manual_seed(1)


class BiLSTMCRF(nn.Module):
    def __init__(self, hidden_dim, num_tags, model_path, device):
        super(BiLSTMCRF, self).__init__()
        embedding_dim = 768
        self.num_tags = num_tags
        self.hidden_dim = hidden_dim
        self.device = device

        self.bert = AutoModel.from_pretrained(model_path)
        self.bilstm = nn.LSTM(embedding_dim, self.hidden_dim // 2,
                              num_layers=1,
                              bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)

    def init_lstm_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_tags, batch_size, self.hidden_dim,
                         requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_tags, batch_size, self.hidden_dim,
                         requires_grad=True).to(self.device)
        return h0, c0

    def forward(self, token_ids, attention_mask, labels=None):
        bert_embeddings = self.bert(input_ids=token_ids,
                                    attention_mask=attention_mask)[0]
        bilstm_embeddings, state = self.bilstm(bert_embeddings)
        linear_embeddings = self.hidden2tag(bilstm_embeddings)

        logits = self.crf.decode(linear_embeddings, mask=attention_mask.bool())

        if labels is None:
            return logits

        loss = -self.crf(linear_embeddings, labels, mask=attention_mask.bool(),
                         reduction="mean")
        return logits, loss
