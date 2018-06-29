#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import os
file_path = 'intent_corpuscontent.txt'
a_file = open(file_path)
index = 1
out_path = 'qa.txt'
out = open(out_path,'w+')
for line in a_file:
    name = line.strip().split('\t')[0]
    os.rename(name,str(index))
    index+=1
    newline = line.strip().split('\t')[1]+'\t'+line.strip().split('\t')[2]+'\n'
    out.write(newline)
a_file.close()
out.close()

