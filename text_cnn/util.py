#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import sys
import re
import time
import logging
from io import StringIO
import json
import pickle
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose

logger = logging.getLogger("rnnlog")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def read_conll(fstream):
    """
    读取CoNLL格式的数据，返回一个大的列表；
    列表里是所有句子，每个句子表示为集合（[词1, 词2, ... , 词n], [标签1， 标签2, ... 标签n]）；
    词和标签均为字符串。
    """

    ret = []
    current_toks, current_lbls = [], []

    for line in fstream:
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
            current_toks, current_lbls = [], []
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(
                line)
            tok, lbl = line.split("\t")
            current_toks.append(tok)
            current_lbls.append(lbl)
    if len(current_toks) > 0:
        assert len(current_toks) == len(current_lbls)
        ret.append((current_toks, current_lbls))
    return ret


def write_conll(fstream, data):
    """
    写入数据为CoNLL格式
    data：[(tokens), (labels), (predictions)]
    tokens、labels、predictions是字符串列表
    """
    for cols in data:
        for row in zip(*cols):
            fstream.write("\t".join(row))
            fstream.write("\n")
        fstream.write("\n")


def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    给定词列表vocab和向量列表vector，将两者一一对应起来；
    前提是两者本身是顺序对应的。
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = array(list(map(float, vector.split())))

    return ret


def window_iterator(seq, n=1, beg="<s>", end="</s>"):
    """遍历
    Iterates through seq by returning windows of length 2n+1
    """
    for i in range(len(seq)):
        l = max(0, i - n)
        r = min(len(seq), i + n + 1)
        ret = seq[l:r]
        if i < n:
            ret = [beg, ] * (n - i) + ret
        if i + n + 1 > len(seq):
            ret = ret + [end, ] * (i + n + 1 - len(seq))
        yield ret


def one_hot(n, y):
    """
    Create a one-hot @n-dimensional vector with a 1 in position @i
    """
    if isinstance(y, int):
        ret = zeros(n)
        ret[y] = 1.0
        return ret
    elif isinstance(y, list):
        ret = zeros((len(y), n))
        ret[np.arange(len(y)), y] = 1.0
        return ret
    else:
        raise ValueError("Expected an int or list got: " + y)


def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """
    表格输出，将二维数据行列输出。
    """
    # Convert data to strings
    data = [["%04.2f" % v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))

    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))

    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret


class ConfusionMatrix(object):
    """
    混淆矩阵
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(
            labels) - 1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_, _ in enumerate(
            self.labels)] for l, _ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__]
                     for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            # 计算评价指标
            acc = (tp + tn) / (tp + tn + fp + fn) if tp > 0 else 0
            prec = tp / (tp + fp) if tp > 0 else 0
            rec = tp / (tp + fn) if tp > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label:  # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average 宏平均 微平均
        tp, fp, tn, fn = micro
        acc = (tp + tn) / (tp + tn + fp + fn) if tp > 0 else 0
        prec = tp / (tp + fp) if tp > 0 else 0
        rec = tp / (tp + fn) if tp > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn) / (tp + tn + fp + fn) if tp > 0 else 0
        prec = tp / (tp + fp) if tp > 0 else 0
        rec = tp / (tp + fn) if tp > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro", "macro", "not-O"], ["label", "acc", "prec", "rec", "f1"])


class Progbar(object):
    """显示进度条
    参考：https://github.com/fchollet/keras/
    参数：
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (
                        k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (
                        k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (
        type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)  # 事实上没有错误
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:
        minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


def print_sentence(output, sentence, labels, predictions):
    # 输出
    spacings = [max(len(sentence[i]), len(labels[i]), len(predictions[i]))
                for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y*: ")
    for token, spacing in zip(labels, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")


def print_sentence2(output, sentence, predictions):
    # 只输出【x】和预测值【y'】
    spacings = [max(len(sentence[i]), len(predictions[i]))
                for i in range(len(sentence))]
    # Compute the word spacing
    output.write("x : ")
    for token, spacing in zip(sentence, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n")

    output.write("y': ")
    for token, spacing in zip(predictions, spacings):
        output.write(token)
        output.write(" " * (spacing - len(token) + 1))
    output.write("\n\n")


def del_space(matched):
    mod = matched.group('value').replace(" ", "")  # 去除实体和标签之间的空格
    return mod


def read_json(fstream, u=None):
    pat = re.compile("<(.*?)>(.*?)</.*?>")  # 标签内的词
    ret = []
    number_record = []

    for line in fstream:
        newline = json.loads(line)
        number_record.append([newline["doc_id"], newline["line_num"]])  # 保存编号
        sentence = newline["sentence"]  # 读取句子（已分词和标注实体）

        words = []
        labels = []
        new_sen = re.sub('(?P<value><(.*?)> (.*?) </.*?>)', del_space, sentence).strip()
        temp = new_sen.split(" ")

        for item in temp:
            finds = re.findall(pat, item)  # 判断是否是实体
            if finds:
                words.append(finds[0][1])  # 实体词
                labels.append(finds[0][0])  # label词
            else:
                words.append(item)  # 非实体词
                labels.append("O")  # 标签-O

        ret.append((words, labels))

    if u:
        output = open('data/number_record.pkl', 'wb')  # 注意，多次调用会覆盖
        pickle.dump(number_record, output)  # 编号存入output文件
        output.close()

    return ret


def write_txt(fstream, data):
    """
    :param fstream:
    :param data:要写入的数据，[([]，[]),] 格式
    :return:
    """
    sen_list = []
    for item in data:
        single_sen = []
        p_words, p_labels = item[0], item[1]

        for w, l in zip(p_words, p_labels):
            if l != "O":
                nw = "<" + l + "> " + w + " </" + l + ">"  # 注意空格
                single_sen.append(nw)
            else:
                single_sen.append(w)
        sen_list.append(single_sen)

    f_num = open("data/number_record.pkl", "rb")
    number_record = pickle.load(f_num)
    temp_dic = {}
    f_num.close()

    for sen, sen_num in zip(sen_list, number_record):
        temp_dic["doc_id"] = sen_num[0]
        temp_dic["line_num"] = sen_num[1]
        temp_dic["sentence"] = " ".join(sen)
        json.dump(temp_dic, fstream, ensure_ascii=False)  # ensure_ascii设为false，保证保存为中文
        fstream.write("\n")
