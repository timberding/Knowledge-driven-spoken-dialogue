import argparse
import json


def read_test(args):
    test_data = []
    with open(args.test_read_path, "r", encoding="utf-8") as file1:
        data = json.load(file1)
        for key, value in data.items():
            for each_message in value:
                test_data.append(each_message["message"]+"\n")

    with open(args.test_save_path, "w", encoding="utf-8") as file2:
        for i in test_data:
            file2.writelines(i)


def read_valid(args):
    save_txt = []
    with open(args.valid_read_path, "r", encoding="utf-8") as file1:
        data = json.load(file1)
        for value in data:
            for each in value["messages"]:
                save_txt.append(each["message"] + "\n")

    with open(args.valid_save_path, "w", encoding="utf-8") as file2:
        for i in save_txt:
            file2.writelines(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_read_path',
                        help='test Path to the input file',
                        required=True)
    parser.add_argument('--test_save_path',
                        help='test to the output file',
                        required=True)
    parser.add_argument('--valid_read_path',
                        help='valid Path to the input file',
                        required=True)
    parser.add_argument('--valid_save_path',
                        help='valid Path to the output file',
                        required=True)

    arg = parser.parse_args()
    read_test(arg)
    read_valid(arg)
