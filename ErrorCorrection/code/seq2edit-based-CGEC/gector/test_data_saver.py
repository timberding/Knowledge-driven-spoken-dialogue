import argparse
import json


def save_test(args):
    final = []
    with open(args.test_read_path, "r", encoding="utf-8") as file1:
        data = file1.readlines()
        for i in data:
            final.append(i.strip().split("\t")[2])
    index = 0

    with open(args.test_original_path, "r", encoding="utf-8") as file2:
        data = json.load(file2)
        for key, value in data.items():
            for each_message in value:
                each_message["message"] = final[index]
                index += 1

    with open(args.test_save_path, "w", encoding="utf-8") as file3:
        json.dump(data, file3, indent=4, ensure_ascii=False)


def save_valid(args):
    final = []
    with open(args.valid_read_path, "r", encoding="utf-8") as file1:
        data = file1.readlines()
        for i in data:
            final.append(i.strip().split("\t")[2])

    index = 0
    with open(args.valid_original_path, "r", encoding="utf-8") as file2:
        data = json.load(file2)
        for value in data:
            for each in value["messages"]:
                each["message"] = final[index]
                index += 1

    with open(args.valid_save_path, "w", encoding="utf-8") as file3:
        json.dump(data, file3, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_read_path',
                        help='test Path to the input file',
                        required=True)

    parser.add_argument('--test_original_path',
                        help='test original path to the input file',
                        required=True)

    parser.add_argument('--test_save_path',
                        help='test to the output file',
                        required=True)

    parser.add_argument('--valid_read_path',
                        help='valid Path to the input file',
                        required=True)

    parser.add_argument('--valid_original_path',
                        help='valid original path to the input file',
                        required=True)

    parser.add_argument('--valid_save_path',
                        help='valid Path to the output file',
                        required=True)

    arg = parser.parse_args()

    save_test(arg)

    save_valid(arg)