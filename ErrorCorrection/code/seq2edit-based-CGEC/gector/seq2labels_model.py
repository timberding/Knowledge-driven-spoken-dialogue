# -*- coding: utf-8

import os
from typing import *

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, tiny_value_of_dtype
from gector.seq2labels_metric import Seq2LabelsMetric
from overrides import overrides
from torch.nn.modules.linear import Linear
from allennlp.nn import util


@Model.register("seq2labels")
class Seq2Labels(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 model_dir: str = "",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 hidden_layers: int = 0,
                 hidden_dim: int = 512,
                 cuda_device: int = 0,
                 dev_file: str = None,
                 logger=None,
                 vocab_path: str = None,
                 weight_name: str = None,
                 save_metric: str = "dev_m2",
                 beta: float = None,
                 ) -> None:

        super(Seq2Labels, self).__init__(vocab, regularizer)
        self.save_metric = save_metric
        self.weight_name = weight_name
        self.cuda_device = cuda_device
        self.device = torch.device("cpu")
        self.label_namespaces = [labels_namespace,
                                 detect_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)
        self.vocab_path = vocab_path
        self.best_metric = 0.0
        self.epoch = 0
        self.model_dir = model_dir
        self.logger = logger
        self.beta = beta
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))
        self.dev_file = dev_file
        self.tag_labels_hidden_layers = []
        self.tag_detect_hidden_layers = []
        input_dim = text_field_embedder.get_output_dim()
        if hidden_layers > 0:
            self.tag_labels_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            self.tag_detect_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            for _ in range(hidden_layers - 1):
                self.tag_labels_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
                self.tag_detect_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_labels_classes)).cuda(self.device)
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_detect_classes)).cuda(self.device)
        else:
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_labels_classes)).to(self.device)
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_detect_classes)).to(self.device)

        self.metric = Seq2LabelsMetric()
        self.UNKID = self.vocab.get_vocab_size("labels") - 2

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        encoded_text = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)

        if self.training:
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, metadata)
            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            logits_labels = ret_train["logits_labels"]
            logits_d = ret_train["logits_d_tags"]
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
            return output_dict

        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, metadata)
        self.train(training_mode)
        logits_labels = ret_train["logits_labels"]
        logits_d = ret_train["logits_d_tags"]
        if labels is not None and d_tags is not None:
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
        return ret_train

    def decode(self, encoded_text: torch.LongTensor = None,
               batch_size: int = 0,
               sequence_length: int = 0,
               mask: torch.LongTensor = None,
               labels: torch.LongTensor = None,
               d_tags: torch.LongTensor = None,
               metadata: List[Dict[str, Any]] = None) -> Dict:
        if self.tag_labels_hidden_layers:
            encoded_text_labels = encoded_text.clone().to(self.device)
            for layer in self.tag_labels_hidden_layers:
                encoded_text_labels = layer(encoded_text_labels)
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text_labels))
            for layer in self.tag_detect_hidden_layers:
                encoded_text = layer(encoded_text)
            logits_d = self.tag_detect_projection_layer(
                self.predictor_dropout(
                    encoded_text))
        else:
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text))
            logits_d = self.tag_detect_projection_layer(
                encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])

        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])

        error_probs = class_probabilities_d[:, :,
                      self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[
            0]

        if self.confidence > 0:
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            offset = torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1)).to(self.device)
            class_probabilities_labels += util.move_to_device(offset, self.device)

        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}

        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics_to_return = self.metric.get_metric(reset)
        if self.metric is not None and not self.training:
            if reset:
                labels_accuracy = float(metrics_to_return['labels_accuracy'].item())
                print('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                labels_accuracy_except_keep = float(metrics_to_return['labels_accuracy_except_keep'].item())
                self.logger.info('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                print('The accuracy of predicting for edit labels except keep label is: ' + str(
                    labels_accuracy_except_keep))
                self.logger.info('The accuracy of predicting for edit labels except keep label is: ' + str(
                    labels_accuracy_except_keep))
                tmp_model_dir = "/".join(self.model_dir.split('/')[:-1]) + "/Temp_Model.th" 
                self.save(tmp_model_dir)
                if self.save_metric == "+labels_accuracy":
                    if self.best_metric <= labels_accuracy:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = labels_accuracy
                        self.save(self.model_dir)
                    print('best labels_accuracy till now:' + str(self.best_metric))
                    self.logger.info('best labels_accuracy till now:' + str(self.best_metric))
                elif self.save_metric == "+labels_accuracy_except_keep":
                    if self.best_metric <= labels_accuracy_except_keep:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = labels_accuracy_except_keep
                        self.save(self.model_dir)
                    print('best labels_accuracy_except_keep till now:' + str(self.best_metric))
                    self.logger.info('best labels_accuracy_except_keep till now:' + str(self.best_metric))
                else:
                    raise NotImplementedError("Wrong metric!")
                self.epoch += 1
                print(f'\nepoch: {self.epoch}')
                self.logger.info(f'epoch: {self.epoch}')

        return metrics_to_return

    def save(self, model_dir):
        with open(model_dir, 'wb') as f:
            torch.save(self.state_dict(), f)
        print("Model is dumped")
        self.logger.info("Model is dumped")
