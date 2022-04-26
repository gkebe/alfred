#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:20:39 2022

@author: gaoussoukebe
"""

import random
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
parser.add_argument('--sample_percent', type=int, help='A number between 0.0 and 1.0')
parser.add_argument('--output', type=str, help='output json file with new train/val/test splits', required=True)
parser.add_argument('--seed', type=float, help='A float to seed the shuffle. Default is 0.5',
                    default=0.5)
args = parser.parse_args()

random.seed(args.seed)
splits_json = args.splits_json

# Opening JSON file with splits
with open(splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

train_split = splits_dict["train"]
k = len(train_split) * args.sample_percent // 100
new_split = random.sample(train_split, k)

splits_dict["train"] = new_split
with open(args.output, "w") as outfile:
    json.dump(splits_dict, outfile, indent=4)