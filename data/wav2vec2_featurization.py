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

import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory where annotations are saved', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
args = parser.parse_args()

def num_steps(dir_path):
    return len([name for name in os.listdir(dir_path) if "step" in name])

def wav2vec2_embed(wav_path):
    speech, _ = sf.read(wav_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.cuda()
    with torch.no_grad():
       hidden_states = model(input_values, output_hidden_states=True).hidden_states
    hidden_states = torch.cat([i for i in hidden_states][-4:]).transpose(0,1).contiguous().view(-1, 3072)
    return torch.mean(hidden_states, dim=0).view(-1)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()

features = {}

# Opening JSON file with splits
with open(args.splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

for split, ann_list in tqdm(splits_dict.items()):
    for ann in tqdm(ann_list):
        speech_dir = os.path.join(args.data_dir, ann["task"], "pp", f"ann_{ann['repeat_idx']}_speech")
        ann_features = {}
        ann_features["lang_goal"] = wav2vec2_embed(f"{speech_dir}/summary.wav").detach().to('cpu')
        ann_features["task_intent"] = wav2vec2_embed(f"{speech_dir}/intention.wav").detach().to('cpu')

        ann_features["lang_instr"] = [wav2vec2_embed(f"{speech_dir}/step_{i}.wav").detach().to('cpu') for i in range(1, num_steps(speech_dir) + 1)]

        features[f"{ann['task']}/ann_{ann['repeat_idx']}.json"] = ann_features


with open(os.path.join(args.data_dir, "wav2vec2_features.pkl"), 'wb') as f:
    pickle.dump(features, f)
