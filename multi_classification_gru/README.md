# Mulit Text Classification with Tensorflow

## Requirements
```
python3
tensorflow >= 1.7
```
## Task
Given a sentence, assign a label according to its' content.
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
python text_rnn.py train
```

3. You can also run an evaluation by:
```
python text_rnn.py evaluate
```
After the program is done, the you can run:
```
python my_evaluate.py
```
to get the result in test set.

4. You can also run a shell by:
```
python text_rnn.py test
```
Then follow the instruction. Hope you enjoy it.

## Reference 
[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf).

## Folder Structure
```
├── data            - this fold contains all the data
│   ├── train
│   ├── dev
│   ├── test
│   ├── vocab
|   ├── vec
├── model           - this fold contains the pkl file to restore
├── text_rnn.py   - main entrance of the project
├── data_util.py    - preprocess the data
├── batch_data.py   - data generator
├── my_evaluate.py  - evaluate the performance of the model in test set   
```

## To do
1. Still need parameters searching.
2. Need structure changing to satisfy parameters chosing.
3. Make codes nicer.
