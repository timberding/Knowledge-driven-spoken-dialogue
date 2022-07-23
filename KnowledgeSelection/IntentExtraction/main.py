import os
import argparse

import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from extractor_model import ExtractorModel
from extractor_dataset import DatasetExtractor


def load_data(datafile, kb):
    data = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(json.loads(line.strip()))
    print(f"length of data: {len(data)}")

    samples = []
    for sample in data:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            if len(subgraph) == 0:
                continue
            text1 = query.replace(entity, "ne")
            for attr in attrs:
                text2 = attr
                for key in subgraph:
                    if key != attr:
                        text3 = key
                        samples.append([text1, text2, text3])
    print(f"length of sample: {len(samples)}")
    return samples


def change_data_format(query, entity2attr, kb, samples):
    for entity, attrs in entity2attr.items():
        subgraph = kb.get(entity, {})
        text1 = query.replace(entity, "ne")
        for attr in attrs:
            text2 = attr
            for key in subgraph:
                if key == attr:
                    continue
                text3 = key
                samples.append([text1, text2, text3])


def load_kb(kbfile):
    kb = {}
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        kb[entity] = {}
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "Information":
                rel = "简介"
            if rel not in kb.get(entity):
                kb.get(entity)[rel] = []
            kb.get(entity)[rel].append(str(tail))
    print(f"length of kb: {len(kb)}")
    return kb


def load_dev_data(datafile, kb):
    samples = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            samples.append(json.loads(line.strip()))
    print(f"length of data: {len(samples)}")

    data = []
    for sample in samples:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            text1 = query.replace(entity, "ne")

            _data = {"text1": text1, "text2": [], "labels": []}
            for attr in subgraph:
                text2 = attr
                _data.get("text2").append(text2)
                if attr in attrs:
                    _data.get("labels").append(1)
                else:
                    _data.get("labels").append(0)
            data.append(_data)

    print(f"length of sample: {len(data)}")
    return data

def train(args):
    pretrain_model_path = args.pretrain_model_path
    save_model_path = args.save_model_path

    max_seq_len = args.max_seq_len
    gpu = args.gpu
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs

    device = torch.device(gpu)
    print("Loading dataset...")

    kb = load_kb(args.kb_file)
    train_data = load_data(args.train_file, kb)
    train_dataset = DatasetExtractor(train_data, max_seq_len,
                                     args.pretrain_model_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False)

    dev_data = load_dev_data(args.dev_file, kb)

    print('Creating model...')
    model = ExtractorModel(device=device, model_path=args.pretrain_model_path)
    print('Model created!')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    best_score = -float("inf")
    not_up_epoch = 0

    model.zero_grad()
    for epoch in range(nb_epochs):
        model.train()
        loader = tqdm(train_dataloader, total=len(train_dataloader),
                      unit="batches")
        running_loss = 0
        for i_batch, data in enumerate(loader):
            model.zero_grad()
            token_ids = data["input_ids"].view(-1, max_seq_len).to(device)
            attention_mask = data["attention_mask"].view(-1, max_seq_len).to(
                device)
            token_type_ids = data["token_type_ids"].view(-1, max_seq_len).to(
                device)
            loss = model(
                token_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(
                Loss=running_loss / ((i_batch + 1) * batch_size),
                Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
        score = validate(device=device, model=model, dev_data=dev_data,
                         dataset=train_dataset)
        if score > best_score + 0.0001:
            best_score = score
            not_up_epoch = 0
            print(
                'Validation accuracy %f increased from previous epoch, '
                'save best_model' % score)
            torch.save(model.state_dict(),
                       os.path.join(save_model_path, "best_model.pt"))
        else:
            not_up_epoch += 1
            if not_up_epoch > 100:
                print(
                    f"Corrcoef didn't up for %s batch, early stop!"
                    % not_up_epoch)
                break


def validate(device, model, dev_data, dataset):
    model.eval()
    score = 0
    acc = 0
    total = len(dev_data)
    for sample in dev_data:
        text1 = sample.get("text1")
        text2 = sample.get("text2")
        labels = sample.get("labels")

        data1 = dataset.sequence_tokenizer(
            [text1])

        seq_vec1 = model.seq2vec(token_ids=data1["input_ids"].to(device),
                                 attention_mask=data1["attention_mask"].to(
                                     device),
                                 token_type_ids=data1["token_type_ids"].to(
                                     device))

        data2 = dataset.sequence_tokenizer(
            text2)

        seq_vec2 = model.seq2vec(token_ids=data2["input_ids"].to(device),
                                 attention_mask=data2["attention_mask"].to(
                                     device),
                                 token_type_ids=data2["token_type_ids"].to(
                                     device))

        seq_vec1 = seq_vec1 / (seq_vec1 ** 2).sum(
            axis=1, keepdims=True) ** 0.5
        seq_vec2 = seq_vec2 / (seq_vec2 ** 2).sum(
            axis=1, keepdims=True) ** 0.5
        similarity = (seq_vec1 * seq_vec2).sum(axis=-1)
        index = np.argmax(similarity)

        if labels[index] == 1:
            acc += 1

    print(acc / total)
    return acc / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--kb_file', type=str, default='')
    parser.add_argument('--pretrain_model_path', type=str, default='')
    parser.add_argument('--save_model_path', type=str, default='')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=100)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "dev":
        pass
    else:
        pass


if __name__ == "__main__":
    main()
