import os
import json


class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None):
        path = os.path.join(os.path.abspath(dataroot))

        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        trains_file = os.path.join(path, dataset, 'data.json')
        self.trains_data = []
        with open(trains_file, "r") as f:
            data = json.load(f)
        for dialog in data:
            if "response" not in dialog.keys() or "history" not in dialog.keys() \
                or "knowledge" not in dialog.keys() or "dialog_id" not in dialog.keys():
                raise ValueError('error in data file: %s' % (dataset))
            self.trains_data.extend(dialog)

    def __iter__(self):
        for dialog in self.trains_data:
            history = dialog["history"]
            response = dialog["response"]
            knowledge = dialog["knowledge"]
            dialog_id = dialog["dialog_id"]
            yield (history, response, knowledge, dialog_id)

    def __len__(self, ):
        return len(self.trains_data)
