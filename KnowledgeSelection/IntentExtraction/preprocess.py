import os
import json
import argparse


def load_data(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    output = open(outputfile, 'w', encoding='utf-8')
    for sample in data:
        messages = sample.get("messages")
        previous_message = messages[0].get("message")
        context = [previous_message]
        prev_entities = []
        for i in range(1, len(messages)):
            message = messages[i].get("message")
            context.append(message)
            if "attrs" in messages[i]:
                attrs = messages[i].get("attrs")
                qsample = dict(question=previous_message, answer=message,
                               knowledge=attrs, context=context,
                               prev_entities=list(set(prev_entities)))
                if previous_message.endswith("？"):
                    output.write(
                        json.dumps(qsample, ensure_ascii=False) + "\n")
                for attr in attrs:
                    if isinstance(attr.get("name"), list):
                        prev_entities.extend(attr.get("name"))
                    else:
                        prev_entities.append(attr.get("name"))
            previous_message = message
    output.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default="")
    parser.add_argument('--output_path', type=str,
                        default="")
    args = parser.parse_args()

    train_file = os.path.join(args.input_path, "train.json")
    output_train_file = os.path.join(args.output_path, "extractor_train.json")
    load_data(train_file, output_train_file)

    valid_file = os.path.join(args.input_path, "valid.json")
    output_valid_file = os.path.join(args.output_path, "extractor_valid.json")
    load_data(valid_file, output_valid_file)


if __name__ == "__main__":
    main()
