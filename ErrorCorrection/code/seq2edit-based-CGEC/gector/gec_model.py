"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time
import torch
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn import util
from pypinyin import lazy_pinyin
from gector.seq2labels_model import Seq2Labels
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class GecBERTModel(object):
    def __init__(self,
                 vocab_path=None,
                 vocab=None,
                 model_paths=None,
                 weights_names=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 log=False,
                 iterations=3,
                 min_probability=0.0,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 cuda_device=0
                 ):

        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(
            model_paths)
        self.cuda_device = int(cuda_device)
        self.device = torch.device(
            "cuda:" + str(cuda_device) if self.cuda_device >= 0 and torch.cuda.is_available() else "cpu")

        self.max_len = max_len
        self.min_len = min_len
        self.min_probability = min_probability
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path) if vocab_path else vocab
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles
        self.indexers = []
        self.models = []

        for model_path, weights_name in zip(model_paths, weights_names):
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            self.indexers.append(self._get_indexer(weights_name))
            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name),
                               confidence=self.confidence,
                               cuda_device=self.cuda_device,
                               ).to(self.device)

            pretrained_dict = torch.load(model_path,
                                         map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def predict(self, batches):
        with torch.cuda.device(self.cuda_device):
            t11 = time()
            predictions = []
            for batch, model in zip(batches, self.models):
                batch = util.move_to_device(batch.as_tensor_dict(), self.cuda_device if torch.cuda.is_available() else -1)
                with torch.no_grad():
                    prediction = model.forward(**batch)
                predictions.append(prediction)

            preds, idx, error_probs, d_tags_idx = self._convert(predictions)
            t55 = time()
            if self.log:
                print(f"Inference time {t55 - t11}")
            return preds, idx, error_probs, d_tags_idx

    def get_token_action(self, index, prob, sugg_token):

        start_pos = 0
        end_pos = 0
        """Get lost of suggested actions for token."""

        if prob < self.min_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    @staticmethod
    def _get_embbeder(weight_name):

        bert_token_emb = PretrainedTransformerEmbedder(model_name=weight_name, last_layer_only=True,
                                                       train_parameters=False)
        token_embedders = {'bert': bert_token_emb}
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=token_embedders)
        return text_field_embedder

    def _get_indexer(self, weight_name):

        bert_token_indexer = PretrainedTransformerIndexer(model_name=weight_name, namespace="bert")
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):

        with torch.cuda.device(self.cuda_device):
            seq_lens = [len(sequence) for sequence in token_batch if sequence]
            if not seq_lens:
                return []
            max_len = min(max(seq_lens), self.max_len)
            batches = []
            for indexer in self.indexers:
                batch = []
                for sequence in token_batch:
                    tokens = sequence[:max_len]
                    tokens = [Token(token) for token in ['$START'] + tokens]
                    batch.append(Instance({'tokens': TextField(tokens, indexer)}))
                batch = Batch(batch)
                batch.index_instances(self.vocab)
                batches.append(batch)
            return batches

    def _convert(self, data):

        all_class_probs = torch.zeros_like(
            data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(
            data[0]['max_error_probability'])
        d_tags_class_probs = torch.zeros_like(
            data[0]['class_probabilities_d_tags'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            d_tags_class_probs += weight * output['class_probabilities_d_tags'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        d_tags_idx = torch.max(d_tags_class_probs,dim=-1)[1]
        max_vals = torch.max(all_class_probs,dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist(), d_tags_idx.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):

        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs, all_d_tags_idxs,
                          max_len=50):

        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob, d_tags_idxs in zip(batch,
                                                           all_probabilities,
                                                           all_idxs,
                                                           error_probs,
                                                           all_d_tags_idxs):
            length = min(len(tokens), max_len)
            edits = []


            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):

                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]

                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')
                action = self.get_token_action(i, probabilities[i], sugg_token)
                if not action: continue
                edits.append(action)

            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch):
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in
                           range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs, d_tags_idxs = self.predict(
                sequences)

            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs, d_tags_idxs, self.max_len)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100 * len(pred_ids) / batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break

        return final_batch, total_updates
