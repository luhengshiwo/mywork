#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import numpy as np
import pandas as pd
from sklearn import metrics
train = pd.read_csv('data/train.csv',sep='\t',header=None)
labels = train.iloc[:,3]
print(labels.value_counts())