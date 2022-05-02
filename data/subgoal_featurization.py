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

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

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

for split, ann_list in tqdm(splits_dict.items()):
    for ann in tqdm(ann_list):
        if ann["task"] in features:
            continue
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_0.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            pddl_plan = ann_json["plan"]["high_pddl"]
        subgoal_features = {}
        for step in pddl_plan:
            subgoal_utterance = step["discrete_action"]["action"]
            for subgoal_arg in step["discrete_action"]["args"]:
                subgoal_utterance += f" {subgoal_arg}"
            subgoal_features[step["high_idx"]] = proc_subgoal(subgoal_utterance)
        features[ann['task']] =subgoal_features


with open(os.path.join(args.data_dir, "subgoal_features.pkl"), 'wb') as f:
    pickle.dump(features, f)
