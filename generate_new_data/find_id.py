#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

__author__ = 'luheng'

import json
domian = '贷款'
file_path = 'trade_map.json'
companylist = []
with open(file_path,'r') as load_f:
    load_dict = json.load(load_f)
for company in load_dict:
    if company['trade'] == domian:
        companylist.append(company['cfg_name'])
print(companylist)

