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
import random

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory where annotations are saved', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
parser.add_argument('--seed', type=float, help='random seed is 71',
                    default=71)
args = parser.parse_args()

random.seed(args.seed)
document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])


def proc_subgoal(t):
    t = t.strip()
    if not t:
        sembeddings = model.encode("")
        return torch.zeros_like(sembeddings)
    sembeddings = model.encode(t)
    return sembeddings


# Opening JSON file with splits
with open(args.splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

options = {}
subgoal_id = {}
action_id = {}
arg_id = {}
subgoal_embeddings = {}


for split, ann_list in tqdm(splits_dict.items()):
    if "test" in split:
        continue
    for ann in tqdm(ann_list):
        if ann["repeat_idx"] != 0:
            continue
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_0.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            action_high_seq = ann_json["num"]["action_high"]
            pddl_plan = ann_json["plan"]["high_pddl"]
        for step in pddl_plan:
            action = step["discrete_action"]["action"]
            if action not in options:
                options[action] = {}
                action_id[action] = action_high_seq[step["high_idx"]]["action"]
            action_args = []
            for i, subgoal_arg in enumerate(step["discrete_action"]["args"]):
                if i not in options[action]:
                    options[action][i] = []
                if subgoal_arg not in options[action][i]:
                    options[action][i].append(subgoal_arg)
                if subgoal_arg not in arg_id:
                    arg_id[subgoal_arg] = action_high_seq[step["high_idx"]]["action_high_args"][i]
                action_args.append(subgoal_arg)
            subgoal_name = action
            if len(action_args):
                subgoal_name += " " + " ".join(action_args)
            if subgoal_name not in subgoal_id:
                i = step["high_idx"]
                subgoal_id[subgoal_name] = int(str(action_id[action]) + "".join([str(arg_id[j]) for j in action_args]))
                subgoal_embeddings[subgoal_id[subgoal_name]] = proc_subgoal(subgoal_name)

features = {"subgoal_id":subgoal_id, "subgoal_embeddings":subgoal_embeddings}
for split, ann_list in tqdm(splits_dict.items()):
    if "test" in split:
        continue
    for ann in tqdm(ann_list):
        if ann["repeat_idx"] != 0:
            continue
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_0.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            action_high_seq = ann_json["num"]["action_high"]
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
            chosen_num_args = 0
            if options[chosen_action]:
                chosen_num_args = max(list(options[chosen_action].keys()))
            chosen_args = []
            for i, subgoal_arg in enumerate(step["discrete_action"]["args"]):
                subgoal_pos += f" {subgoal_arg}"
                if i not in options[chosen_action]:
                    continue
                arg_options = options[chosen_action][i]
                if subgoal_arg in arg_options:
                    if not (same and i == chosen_num_args - 1) and (random.choice([0] * 3 + [1]) == 1):
                        arg_options = [subgoal_arg]
                    else:
                        same = False
                        arg_options.remove(subgoal_arg)
                if len(arg_options):
                    chosen_arg = random.choice(arg_options)
                    chosen_args.append(chosen_arg)
                    subgoal_neg += f" {chosen_arg}"
                else:
                    subgoal_neg = "NoOp"

            if subgoal_neg not in subgoal_id:
                i = step["high_idx"]
                subgoal_id[subgoal_neg] = int(str(action_id[chosen_action]) + "".join([str(arg_id[j]) for j in chosen_args]))
                subgoal_embeddings[subgoal_id[subgoal_neg]] = proc_subgoal(subgoal_neg)

            subgoal_features[step["high_idx"]] = {"pos": subgoal_id[subgoal_pos],
                                                  "neg": subgoal_id[subgoal_neg]}
        features[ann['task']] =subgoal_features




with open(os.path.join(args.data_dir, "subgoal_features_.pkl"), 'wb') as f:
    pickle.dump(features, f)
