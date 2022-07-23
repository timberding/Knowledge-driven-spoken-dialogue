import argparse
import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ner_model import BiLSTMCRF
from ner_dataset import NERDataset
from ner_metrics import NERMetric


def load_data(datafile):
    with open(datafile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    print(f"length of data: {len(data)}")
    samples = []
    for sample in data:
        text = sample.get("text")
        labels = sample.get("labels")
        samples.append([text, labels])
    return samples


def load_tagid(tagfile):
    tag2idx, idx2tag = {}, {}
    with open(tagfile, 'r', encoding='utf-8') as fin:
        for line in fin:
            tag, idx = line.strip().split("\t")
            tag2idx[tag] = int(idx)
            idx2tag[int(idx)] = tag
    return tag2idx, idx2tag


def train(args):
    train_file = args.train_file
    dev_file = args.dev_file
    tag_file = args.tag_file
    pretrain_model_path = args.pretrain_model_path
    save_model_path = args.save_model_path

    max_seq_len = args.max_seq_len
    gpu = args.gpu
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs
    validate_every = args.validate_every
    patience = args.patience

    device = torch.device(gpu)
    print("Loading dataset...")
    tag2idx, idx2tag = load_tagid(tag_file)
    train_data = load_data(train_file)
    train_dataset = NERDataset(train_data, tag2idx, max_seq_len,
                               pretrain_model_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True)

    dev_data = load_data(dev_file)
    dev_dataset = NERDataset(dev_data, tag2idx, max_seq_len,
                             pretrain_model_path)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,
                                collate_fn=dev_dataset.collate_fn,
                                shuffle=True)
    ner_metrics = NERMetric(idx2tag)

    print('Creating model...')
    model = BiLSTMCRF(hidden_dim=200, num_tags=len(tag2idx),
                      model_path=pretrain_model_path, device=device)
    print('Model created!')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer)
    optimizer.zero_grad()

    best_score = -float("inf")
    no_update = 0

    model.zero_grad()
    for epoch in range(nb_epochs):
        print('Training epoch %d' % epoch)
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                print('Training step...')
                model.train()
                loader = tqdm(train_dataloader, total=len(train_dataloader),
                              unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    token_ids = a[0].to(device)
                    attention_mask = a[1].to(device)
                    labels = a[2].to(device)
                    logit, loss = model(token_ids=token_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(
                        Loss=running_loss / ((i_batch + 1) * batch_size),
                        Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                print(
                    'loss is %f' % (running_loss / (len(loader) * batch_size)))
                # scheduler.step(metrics=running_loss, epoch=epoch)
            else:
                print('Valid step...')
                model.eval()
                eps = 0.0001
                score = validate(device=device, model=model,
                                 dev_dataloader=dev_dataloader,
                                 ner_metrics=ner_metrics)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    print(
                        'Validation accuracy %f increased from previous epoch'
                        % score)
                    torch.save(model.state_dict(),
                               os.path.join(save_model_path, "best_model.pt"))
                    continue
                elif (score < best_score + eps) and (no_update < patience):
                    no_update += 1
                    print(
                        "Validation accuracy decreases to %f from %f, "
                        "%d more epoch to check" % (
                            score, best_score, patience - no_update))
                elif no_update == patience:
                    print(
                        "Model has exceed patience. Saving model and exiting")
                    exit()
                torch.save(model.state_dict(),
                           os.path.join(save_model_path, "x.pt"))


def validate(device, model, dev_dataloader, ner_metrics):
    model.eval()
    for a in dev_dataloader:
        token_ids = a[0].to(device)
        attention_mask = a[1].to(device)
        labels = a[2].numpy().tolist()
        logit = model(token_ids=token_ids,
                      attention_mask=attention_mask)
        prediction = logit
        ner_metrics.update_batch(prediction, labels)
    metrics = ner_metrics.get_metric(reset=True)
    print(metrics)
    return metrics["f1"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--tag_file', type=str, default='')
    parser.add_argument('--pretrain_model_path', type=str, default='')
    parser.add_argument('--save_model_path', type=str, default='')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "dev":
        pass
    else:
        pass


if __name__ == "__main__":
    main()
