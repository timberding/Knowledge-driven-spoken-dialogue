# digix2022_Knowledge-driven-spoken-dialogue

## Baseline
Baseline has three modules: text error correction, knowledge selection and generative model incorporating knowledge:
1) Text Error Correction Module: Correcting grammar error of input text by replace, delete, insert and change order
2) Knowledge Selection Module: Identifying entities from dialogue history ,and extracting attributes of entities that involved in current utterance; Selecting related triples from knowledge base files according to the entites and attributes.
3) Generative Model Incorporating Knowledge Module: Incorporating related knowledge triples into conversation and generate reply text with knowledge.

## score
|precision|recall|F1|bleu-1|bleu-2|generate-F1|score|
|---------|:---------:|:--------:|:---------:|:--------:|:---------:|:--------:|
|0.5015|0.4929|0.4856|0.2869|0.2231|0.397|1.0789|
