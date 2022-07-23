
class NERMetric():
    def __init__(self, idx2tag):
        self.idx2tag = idx2tag
        self._precision = 0.0
        self._recall = 0.0
        self._f1 = 0.0
        self._count = 0.0

    def update(self, pred_labels, true_labels):
        self._count += 1
        true_entities = self.get_entity(true_labels)
        pred_entities = self.get_entity(pred_labels)
        precision, recall, f1 = self.calculator(pred_entities,
                                                true_entities)
        self._precision += precision
        self._recall += recall
        self._f1 += f1

    def update_batch(self, predictions, golds):
        batch_size = len(golds)
        self._count += batch_size
        for pred_labels, true_labels in zip(predictions, golds):
            true_entities = self.get_entity(true_labels)
            pred_entities = self.get_entity(pred_labels)
            precision, recall, f1 = self.calculator(pred_entities,
                                                    true_entities)
            self._precision += precision
            self._recall += recall
            self._f1 += f1

    def get_metric(self, reset: bool = False):
        precision, recall, f1 = 0.0, 0.0, 0.0
        if self._count != 0.0:
            precision = self._precision / self._count
            recall = self._recall / self._count
            f1 = self._f1 / self._count
        if reset:
            self._precision = 0.0
            self._recall = 0.0
            self._f1 = 0.0
            self._count = 0.0
        return {"precision": precision,
                "recall": recall,
                "f1": f1}

    def get_entity(self, label_ids):
        labels = [self.idx2tag[v] for v in label_ids]
        entities = []
        etype, start, end = "", -1, -1
        for index, tag in enumerate(labels):
            if "-" not in tag:
                _type = ""
            else:
                _type = tag.split("-")[-1]
            if tag.startswith("S"):
                etype = _type
                start = index
                end = index
                entities.append([etype, start, end])
                etype, start, end = "", -1, -1
            elif tag.startswith("B"):
                etype = _type
                start = index
            elif tag.startswith("I"):
                if _type == etype:
                    end = index
                if index == len(labels) - 1:
                    entities.append([etype, start, end])
            else:
                if end != -1:
                    entities.append([etype, start, end])
                etype, start, end = "", -1, -1
        return entities

    def calculator(self, pred_entities, true_entities):
        tp = 0
        for pred in pred_entities:
            if pred in true_entities:
                tp += 1
        precision = tp / len(pred_entities) if len(
            pred_entities) > 0 else 0.0
        recall = tp / len(true_entities) if len(true_entities) > 0 else 0.0
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
