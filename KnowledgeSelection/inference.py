import jieba
import jieba.posseg as pseg

import numpy as np
import json
import argparse
from collections import Counter

from NameEntityRecognition.ner_infere import NERInfere
from IntentExtraction.extractor_infere import IntentInfere


class KnowledgeSelection():
    def __init__(self, args):
        self.kb, self.entity_mapping = self.load_kb(args.kb_file)
        self.ner_infere = NERInfere(args.gpu, args.tag_file,
                                    args.ner_pretrain_model_path,
                                    args.ner_save_model_path,
                                    args.ner_max_seq_len)

        self.intent_infere = IntentInfere(args.gpu,
                                          args.extractor_pretrain_model_path,
                                          args.extractor_save_model_path,
                                          args.extractor_max_seq_len)

        self.kb_enitites = list(self.kb.keys())
        self.idf = {}

        for word in self.kb_enitites:
            jieba.add_word(word, 100, "entity")

    def load_kb(self, kbfile):
        kb = {}
        entity_mapping = {}
        with open(kbfile, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for entity in data:
            if "（" in entity:
                new_entity = entity.split("（")[0]
                entity_mapping[new_entity] = entity
            kb[entity] = {}
            for attr in data.get(entity):
                head, rel, tail = attr
                if rel == "Information":
                    rel = "简介"
                if rel not in kb.get(entity):
                    kb.get(entity)[rel] = []
                if tail not in kb.get(entity)[rel]:
                    kb.get(entity)[rel].append(tail)
        print(f"length of kb: {len(kb)}")
        return kb, entity_mapping

    def get_idf(self, sentences):
        idf = Counter()
        for sent in sentences:
            words = jieba.lcut(sent)
            words = list(set(words))
            idf.update(words)
        for key in idf:
            idf[key] = len(sentences) / idf[key]
        return idf

    def get_entity_by_jieba(self, context):
        candidates = []
        for seq in context:
            words = pseg.cut(seq)
            for (word, pos) in words:
                if pos == "entity":
                    candidates.append(word)

        pred_words = {}
        for word in candidates:
            if word not in self.kb_enitites:
                continue
            s = self.idf.get(word, 5)
            pred_words[word] = s

        pred_words = dict(
            sorted(pred_words.items(), key=lambda x: x[1], reverse=True))
        return list(pred_words.keys())[:1]

    def get_entities(self, context):
        entities = Counter()
        for sent in context:
            pred_entities = self.ner_infere.ner(sent)
            for pred in pred_entities:
                if pred == "":
                    continue
                if pred in self.kb_enitites:
                    entities.update([pred])
        entities = list(entities.keys())
        if len(entities) == 0:
            entities = self.get_entity_by_jieba(context)
        return entities

    def get_intent(self, entities, query):
        candidates = []
        global_entities = []
        for entity in entities:
            attrs = list(self.kb.get(entity, {}).keys())
            candidates.extend(attrs)
            global_entities.extend([entity] * len(attrs))
        if len(candidates) == 0:
            return None, None
        pred_score = self.intent_infere.text_smiliary([query], candidates)
        index = np.argmax(pred_score)
        pred_intent = candidates[index]
        pred_entity = global_entities[index]
        return pred_intent, pred_entity

    def get_pred_knowledge(self, entity, intent):
        if entity is None:
            return []
        pred_knowledge = []
        if entity not in self.kb:
            return []
        if intent not in self.kb.get(entity):
            print(f"{intent} not in {self.kb.get(entity)}")
            return []

        for value in self.kb.get(entity)[intent]:
            if intent == "简介":
                intent = "Information"
            known = {"name": entity, "attrname": intent, "attrvalue": value}
            pred_knowledge.append(known)
        return pred_knowledge

    def _match(self, gold_knowledge, pred_knowledge):
        result = []
        for pred in pred_knowledge:
            matched = False
            for gold in gold_knowledge:
                if isinstance(pred["attrvalue"], list):
                    pred_attrvalue = " ".join(sorted(pred["attrvalue"]))
                else:
                    pred_attrvalue = pred["attrvalue"]
                if isinstance(gold["attrvalue"], list):
                    gold_attrvalue = " ".join(sorted(gold["attrvalue"]))
                else:
                    gold_attrvalue = gold["attrvalue"]
                if pred['name'] == gold['name'] and pred['attrname'] == gold[
                    'attrname'] and pred_attrvalue == gold_attrvalue:
                    matched = True
            result.append(matched)
        return result

    def calu_knowledge_selection(self, gold_knowledge, pred_knowledge):
        if len(gold_knowledge) == 0 and len(pred_knowledge) == 0:
            return 1.0, 1.0, 1.0

        precision, recall, f1 = 0.0, 0.0, 0.0
        relevance = self._match(gold_knowledge, pred_knowledge)
        if len(relevance) == 0 or sum(relevance) == 0:
            return precision, recall, f1

        tp = sum(relevance)
        precision = tp / len(pred_knowledge) if len(
            pred_knowledge) > 0 else 0.0
        recall = tp / len(gold_knowledge) if len(gold_knowledge) > 0 else 0.0
        if precision == 0 and recall == 0:
            return precision, recall, f1
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def load_valid_data(self, valid_file):
        with open(valid_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        samples = []
        all_messages = []
        for sample in data:
            messages = sample.get("messages")
            previous_message = messages[0].get("message")
            all_messages.append(previous_message)
            context = [previous_message]
            prev_entities = []
            for i in range(1, len(messages)):
                message = messages[i].get("message")
                all_messages.append(message)
                context.append(message)
                if "attrs" in messages[i]:
                    attrs = messages[i].get("attrs")
                    qsample = dict(question=previous_message, answer=message,
                                   knowledge=attrs, context=context,
                                   prev_entities=list(set(prev_entities)))
                    if previous_message.endswith("？"):
                        samples.append(qsample)
                    prev_entities.extend([attr.get("name") for attr in attrs])
                previous_message = message
        self.idf = self.get_idf(all_messages)
        return samples

    def load_test_data(self, test_file):
        with open(test_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        samples = {}
        all_messages = []
        for index in data:
            question = data[index][-1].get("message")
            context = [turn["message"] for turn in data[index]]
            all_messages.extend(context)
            sample = {"question": question, "context": context}
            samples[index] = sample
        self.idf = self.get_idf(all_messages)
        return samples

    def evaluate(self, datafile):
        data = self.load_valid_data(datafile)

        total = len(data)
        metrics = {"p": 0, "r": 0, "f1": 0}
        for sample in data:
            knowledge = sample.get("knowledge")

            context = sample.get("context")
            question = sample.get("question")
            entities = self.get_entities(context)
            for i, entity in enumerate(entities):
                if entity in self.entity_mapping:
                    entities[i] = self.entity_mapping.get(entity)

            intent, entity = self.get_intent(entities, question)

            pred_knowledge = self.get_pred_knowledge(entity, intent)
            p, r, f1 = self.calu_knowledge_selection(knowledge, pred_knowledge)

            metrics["p"] += p
            metrics["r"] += r
            metrics["f1"] += f1

        for key in metrics:
            metrics[key] = metrics.get(key) / total
        print(metrics)
        return metrics

    def test(self, datafile, outputfile):
        data = self.load_test_data(datafile)

        samples = {}
        for index in data:
            question = data.get(index).get("question")
            context = data.get(index).get("context")

            entities = self.get_entities(context)
            for i, entity in enumerate(entities):
                if entity in self.entity_mapping:
                    entities[i] = self.entity_mapping.get(entity)

            intent, entity = self.get_intent(entities, question)

            pred_knowledge = self.get_pred_knowledge(entity, intent)
            sample = {"question": question, "context": context,
                      "attrs": pred_knowledge}
            samples[index] = sample

        with open(outputfile, 'w', encoding='utf-8') as fout:
            json.dump(samples, fout, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        default="valid")

    parser.add_argument('--tag_file', type=str,
                        default="")
    parser.add_argument('--ner_pretrain_model_path', type=str,
                        default="")
    parser.add_argument('--ner_save_model_path', type=str,
                        default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ner_max_seq_len', type=int, default=512)

    parser.add_argument('--extractor_pretrain_model_path', type=str,
                        default="")

    parser.add_argument('--extractor_save_model_path', type=str,
                        default="")
    parser.add_argument('--extractor_max_seq_len', type=int, default=50)

    parser.add_argument('--kb_file', type=str,
                        default="")
    parser.add_argument('--valid_file', type=str,
                        default="")
    parser.add_argument('--test_file', type=str,
                        default="")
    parser.add_argument('--result_file', type=str,
                        default="")

    args = parser.parse_args()

    selector = KnowledgeSelection(args)
    if args.mode == "test":
        selector.test(args.test_file, args.result_file)
    else:
        selector.evaluate(args.valid_file)


if __name__ == "__main__":
    main()
