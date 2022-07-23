# -*- coding: utf-8
import argparse
import logging
import torch
import time
import os

from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.training.optimizers import AdamOptimizer
from gector.datareader import Seq2LabelsDatasetReader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training import GradientDescentTrainer
from allennlp.data.vocabulary import Vocabulary
from gector.seq2labels_model import Seq2Labels
from allennlp.data import allennlp_collate
from torch.utils.data import DataLoader
from allennlp.modules import Embedding
from random import seed


def fix_seed(s):
    torch.manual_seed(s)
    seed(s)


def get_token_indexers(model_name):
    bert_token_indexer = PretrainedTransformerIndexer(model_name=model_name, namespace="bert")
    return {'bert': bert_token_indexer}


def get_token_embedders(model_name, tune_bert=False):
    take_grads = True if tune_bert > 0 else False
    bert_token_emb = PretrainedTransformerEmbedder(model_name=model_name, last_layer_only=True,
                                                   train_parameters=take_grads)
    token_embedders = {'bert': bert_token_emb}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders)
    return text_filed_emd


def build_data_loaders(
        data_set: AllennlpDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        batches_per_epoch=None):
    return PyTorchDataLoader(data_set, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                             collate_fn=allennlp_collate, batches_per_epoch=batches_per_epoch)


def get_data_reader(model_name, max_len,
                    skip_correct=False,
                    skip_complex=0,
                    test_mode=False,
                    tag_strategy="keep_one",
                    broken_dot_strategy="keep",
                    tn_prob=0,
                    tp_prob=1, ):
    token_indexers = get_token_indexers(model_name)
    reader = Seq2LabelsDatasetReader(token_indexers=token_indexers,
                                     max_len=max_len,
                                     skip_correct=skip_correct,
                                     skip_complex=skip_complex,
                                     test_mode=test_mode,
                                     tag_strategy=tag_strategy,
                                     broken_dot_strategy=broken_dot_strategy,
                                     lazy=True,
                                     tn_prob=tn_prob,
                                     tp_prob=tp_prob)
    return reader


def get_model(model_name, vocab,
              tune_bert=False,
              predictor_dropout=0,
              label_smoothing=0.0,
              confidence=0,
              model_dir="",
              log=None):

    token_embs = get_token_embedders(model_name, tune_bert=tune_bert)
    model = Seq2Labels(vocab=vocab,
                       text_field_embedder=token_embs,
                       predictor_dropout=predictor_dropout,
                       label_smoothing=label_smoothing,
                       confidence=confidence,
                       model_dir=model_dir,
                       cuda_device=args.cuda_device,
                       dev_file=args.dev_set,
                       logger=log,
                       vocab_path=args.vocab_path,
                       weight_name=args.weights_name,
                       save_metric=args.save_metric
                       )
    return model


def main(args):
    fix_seed(args.seed)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    logger = logging.getLogger(__file__)
    logger.setLevel(level=logging.INFO)
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    handler = logging.FileHandler(args.model_dir + '/logs_{:s}.txt'.format(str(start_time)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    weights_name = args.weights_name
    reader = get_data_reader(weights_name, args.max_len, skip_correct=bool(args.skip_correct),
                             skip_complex=args.skip_complex,
                             test_mode=False,
                             tag_strategy=args.tag_strategy,
                             tn_prob=args.tn_prob,
                             tp_prob=args.tp_prob)
    train_data = reader.read(args.train_set)
    dev_data = reader.read(args.dev_set)

    default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
    namespaces = ['labels', 'd_tags']
    tokens_to_add = {x: default_tokens for x in namespaces}

    # build vocab
    if args.vocab_path:
        vocab = Vocabulary.from_files(args.vocab_path)
    else:
        vocab = Vocabulary.from_instances(train_data,
                                          min_count={"labels": 5},
                                          tokens_to_add=tokens_to_add)
        vocab.save_to_files(args.vocab_path)

    print("Data is loaded")
    logger.info("Data is loaded")

    model = get_model(weights_name, vocab,
                      tune_bert=args.tune_bert,
                      predictor_dropout=args.predictor_dropout,
                      label_smoothing=args.label_smoothing,
                      model_dir=os.path.join(args.model_dir, args.model_name + '.th'),
                      log=logger)

    device = torch.device("cuda:" + str(args.cuda_device) if int(args.cuda_device) >= 0 else "cpu")
    if args.pretrain:
        pretrained_dict = torch.load(os.path.join(args.pretrain_folder, args.pretrain + '.th'), map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model')
        logger.info('load pretrained model')

    model = model.to(device)
    print("Model is set")
    logger.info("Model is set")

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer = AdamOptimizer(parameters, lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceOnPlateauLearningRateScheduler(optimizer)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    tensorboardWriter = TensorboardWriter(args.model_dir)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=build_data_loaders(train_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                       batches_per_epoch=args.updates_per_epoch),
        validation_data_loader=build_data_loaders(dev_data, batch_size=args.batch_size, num_workers=0, shuffle=False),
        num_epochs=args.n_epoch,
        optimizer=optimizer,
        patience=args.patience,
        validation_metric=args.save_metric,
        cuda_device=device,
        num_gradient_accumulation_steps=args.accumulation_size,
        learning_rate_scheduler=scheduler,
        tensorboard_writer=tensorboardWriter,
        use_amp=True
    )
    print("Start training")
    print('\nepoch: 0')
    logger.info("Start training")
    logger.info('epoch: 0')
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set',
                        help='Path to the train data',
                        required=True)
    parser.add_argument('--dev_set',
                        help='Path to the dev data',
                        required=True)
    parser.add_argument('--model_dir',
                        help='Path to the model dir',
                        required=True)
    parser.add_argument('--model_name',
                        help='The name of saved checkpoint',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.'
                             'If not set then build vocab from data',
                        default="./data/output_vocabulary_chinese_char_hsk+lang8_5")
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=256)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=200)
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=2)
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)
    parser.add_argument('--skip_complex',
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=0)
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-3)
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)
    parser.add_argument('--tp_prob',
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)
    parser.add_argument('--pretrain_folder',
                        help='The name of the pretrain folder.',
                        default=None)
    parser.add_argument('--pretrain',
                        help='The name of the pretrain weights in pretrain_folder param.',
                        default=None)
    parser.add_argument('--cuda_device',
                        help='The number of GPU',
                        default=0)
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=1)
    parser.add_argument('--weights_name',
                        type=str,
                        default="chinese-struct-bert")
    parser.add_argument('--save_metric',
                        type=str,
                        choices=["+labels_accuracy", "+labels_accuracy_except_keep"],
                        default="+labels_accuracy")
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        default=1) 
    args = parser.parse_args()
    main(args)
