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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory where annotations are saved', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
args = parser.parse_args()

def setup_device(gpu_num=0):
    """Setup device."""
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'  # Is there a GPU?
    device = torch.device(device_name)
    return device
#astype = lambda x: np.array(x).astype(np.uint8)


document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])

def proc_sentence(t):
    t = t.strip()
    if not t:
        sentence = flair.data.Sentence("hello", use_tokenizer=True)
        document_embeddings.embed(sentence)
        return torch.zeros_like(sentence.get_embedding())
    sentence = flair.data.Sentence(t, use_tokenizer=True)
    document_embeddings.embed(sentence)
    return sentence.get_embedding()

features = {}

# Opening JSON file with splits
with open(args.splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

for split, ann_list in tqdm(splits_dict.items()):
    for ann in tqdm(ann_list):
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_{ann['repeat_idx']}.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            ann_dict = ann_json["turk_annotations"]["anns"][ann_json["repeat_idx"]]
        ann_features = {}
        ann_features["lang_goal"] = proc_sentence(ann_dict["task_desc"]).detach().to('cpu')
        ann_features["task_intent"] = proc_sentence(ann_dict["task_intent"]).detach().to('cpu')

        ann_features["lang_instr"] = [proc_sentence(step).detach().to('cpu') for step in ann_dict["high_descs"]]
        features[f"{ann['task']}/ann_{ann['repeat_idx']}.json"] = ann_features


with open(os.path.join(args.data_dir, "bert_features.pkl"), 'wb') as f:
    pickle.dump(features, f)
