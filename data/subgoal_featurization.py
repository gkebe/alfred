#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:20:39 2022

@author: gaoussoukebe
"""

from tqdm import tqdm
import numpy as np
import torch
import flair
import pickle
import argparse
import json
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory where annotations are saved', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
args = parser.parse_args()

document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])

def proc_subgoal(t):
    t = t.strip()
    if not t:
        sembeddings = model.encode("")
        return torch.zeros_like(sembeddings)
    sembeddings = model.encode(t)
    return sembeddings

features = {}

# Opening JSON file with splits
with open(args.splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

options = {}

for split, ann_list in tqdm(splits_dict.items()):
    if "test" in split:
        continue
    for ann in tqdm(ann_list):
        if ann["task"] in features:
            continue
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_0.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            pddl_plan = ann_json["plan"]["high_pddl"]
        for step in pddl_plan:
            action = step["discrete_action"]["action"]
            if action not in options:
                options[action] = {}
            for i, subgoal_arg in enumerate(step["discrete_action"]["args"]):
                if i not in options[action]:
                    options[action][i] = []
                if subgoal_arg not in options[action][i]:
                    options[action][i].append(subgoal_arg)

for split, ann_list in tqdm(splits_dict.items()):
    if "test" in split:
        continue
    for ann in tqdm(ann_list):
        if ann["task"] in features:
            continue
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_0.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            pddl_plan = ann_json["plan"]["high_pddl"]
        subgoal_features = {}
        for step in pddl_plan:
            subgoal_pos= step["discrete_action"]["action"]
            same = False
            if random.choice([0]*3 + [1]) == 1:
                subgoal_neg = step["discrete_action"]["action"]
                same = True
            else:
                action_options = list(options.keys())
                action_options.remove(subgoal_pos)
                subgoal_neg = random.choice(action_options)

            chosen_action = subgoal_neg
            chosen_num_args = max(list(options[choosen_action].keys()))
            for i, subgoal_arg in emunerate(step["discrete_action"]["args"]):
                subgoal_pos += f" {subgoal_arg}"
                if i not in options[choosen_action]:
                    continue
                arg_options = options[subgoal_negative][i]
                if subgoal_arg in arg_options:
                    if not (same and i == chosen_num_args - 1) and random.choice([0] * 3 + [1]) == 1:
                        arg_options = [subgoal_arg]
                    else:
                        same = False
                        arg_options.remove(subgoal_arg)

                subgoal_neg += f" {random.choice(arg_options)}"

            print(f"Pos: {subgoal_pos}")
            print(f"Neg: {subgoal_neg}")
            print()
            #subgoal_features[step["high_idx"]] = {"pos": proc_subgoal(subgoal_pos),
                                                  #"neg": proc_subgoal(subgoal_neg)}
        #features[ann['task']] =subgoal_features



with open(os.path.join(args.data_dir, "subgoal_features.pkl"), 'wb') as f:
    pickle.dump(features, f)
