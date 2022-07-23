import os
import json
import argparse


def load_data(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    pairs = []
    for sample in data:
        messages = sample.get("messages")
        context = []
        entities = set()
        for message in messages:
            context.append(message.get("message"))
            attrs = message.get("attrs", [])
            for attr in attrs:
                if isinstance(attr.get("name"), list):
                    entities.update(attr.get("name"))
                else:
                    entities.add(str(attr.get("name")))
        sequence_label(context, entities, pairs)

    with open(outputfile, 'w', encoding='utf-8') as fout:
        json.dump(pairs, fout, ensure_ascii=False)


def sequence_label(context, entities, pairs):
    for sequence in context:
        tags = ["O"] * len(sequence)
        sample = {"raw_text": sequence, "entity": {}}
        for entity in entities:
            if entity in sequence:
                start_index = get_sequence_labels(sequence, entity, tags)
                sample.get("entity")[entity] = start_index

        if "B" in tags or "S" in tags:
            sample = {"text": sequence, "labels": tags}
            pairs.append(sample)


def get_sequence_labels(sequence, entity, tags):
    index = sequence.find(entity)
    start_index = []
    while (index != -1 and index + len(entity) < len(sequence) - 1):
        if tags[index] == "O":
            if len(entity) == 1:
                tags[index] = "S"
            else:
                tags[index] = "B"
            start_index.append(index)
        for i in range(index + 1, index + len(entity)):
            if tags[i] == "O":
                tags[i] = "I"
        index = sequence.find(entity, index + len(entity))
    return start_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default="")
    parser.add_argument('--output_path', type=str,
                        default="")
    args = parser.parse_args()

    train_file = os.path.join(args.input_path, "train.json")
    output_train_file = os.path.join(args.output_path, "ner_train.json")
    load_data(train_file, output_train_file)

    valid_file = os.path.join(args.input_path, "valid.json")
    output_valid_file = os.path.join(args.output_path, "ner_valid.json")
    load_data(valid_file, output_valid_file)


if __name__ == "__main__":
    main()
