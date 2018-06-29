# Seq2seq with Tensorflow

## Requirements
```
python3
tensorflow >= 1.7
```
## Task
Given a sentence, transform to a better sentence.
```

```

## data & preprocess
The data provided in the `data/` directory is a csv file

In `data_util.py` I provide some funtions to process the csv file.

## Usage
This contains several steps:
1. Before you can get started on training the model, you mast run
```
python data_util.py
```

2. After the dirty preprpcessing jobs, you can try running an training experiment with some configurations by:
```
python attention_seq2seq.py train
```

3. You can also run an evaluation by:
```
python attention_seq2seq.py evaluate
```
4. You can also run a shell by:
```
python attention_seq2seq.py test
```
Then follow the instruction. Hope you enjoy it.

## Reference 
[Neural Machine Translation (seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq).
[Sequence to Sequence Learningwith Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).

## Folder Structure
```
├── data            - this fold contains all the data
│   ├── train
│   ├── dev
│   ├── test
│   ├── vocab
|   ├── vec
├── model           - this fold contains the pkl file to restore
├── attention_seq2seq.py   - main entrance of the project
├── data_util.py    - preprocess the data
├── generate.py   - data generator
├── my_evaluate.py  - evaluate the performance of the model in test set   
```

## To do
1. Still need parameters searching.
2. Need structure changing to satisfy parameters chosing.
3. Make codes nicer.
