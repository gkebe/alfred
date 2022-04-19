#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:20:39 2022

@author: gaoussoukebe
"""

import shutil
import json
import os
import argparse
import librosa
import soundfile as sf
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory where annotations are saved', required=True)
parser.add_argument('--speech_dir', type=str, help='directory with speech files', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)

args = parser.parse_args()

# where alfred data is located
data_dir = args.data_dir
# where speech data is located
speech_dir = args.speech_dir

splits_json = args.splits_json

# Opening JSON file with splits
with open(splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

for split, ann_list in tqdm(splits_dict.items()):
    for ann in tqdm(ann_list):
        dest_dir = os.path.join(data_dir, ann["task"], "pp", f"ann_{ann['repeat_idx']}_speech")
        with open(os.path.join(args.data_dir, ann["task"], "pp", f"ann_{ann['repeat_idx']}.json"), 'r') as ann_file:
            ann_json = json.loads(ann_file.read())
            ann_dict = ann_json["turk_annotations"]["anns"][ann_json["repeat_idx"]]
        os.mkdir(dest_dir)
        src_dir = os.path.join(speech_dir, ann["task"].replace("/", "_"), ann_dict["worker_id"])
        for filename in tqdm(os.listdir(src_dir)):
            y, s = librosa.load(os.path.join(src_dir, filename), sr=16000)
            sf.write(os.path.join(dest_dir, filename), y, s)