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

parser = argparse.ArgumentParser()
parser.add_argument('--alfred_data_dir', type=str, help='json_feat_2.1.0 directory from the original Alfred data', required=True)
parser.add_argument('--output_dir', type=str, help='directory where data with new annotations will be saved', required=True)
parser.add_argument('--transcriptions_json', type=str, help='.jsonl file with transcriptions of spoken annotations', required=True)
parser.add_argument('--splits_json', type=str, help='json file with train/val/test splits', required=True)
args = parser.parse_args()

# where alfred data is located
alfred_data_dir = args.alfred_data_dir
# where we are putting alfred data
annotations_dir = args.output_dir

transcriptions_jsonl = args.transcriptions_json

splits_json = args.splits_json

# if where we are moving data already exists delete it
if os.path.exists(annotations_dir):
    shutil.rmtree(annotations_dir)

# create annatations folder
os.mkdir(annotations_dir)

# Opening JSON file of transcripts
with open(transcriptions_json, 'r') as json_file:
    json_list = list(json_file)

# Opening JSON file with splits
with open(splits_json, 'r') as json_file:
    splits_dict = json.loads(json_file.read())

# Get all the videos for which there are annatations
trials = {}
for json_str in json_list:
    result = json.loads(json_str)
    # generate name
    if '-'.join(result['task'].split('-')[:5]) in trials:
        trials['-'.join(result['task'].split('-')[:5])].append(result)
    else:
        trials['-'.join(result['task'].split('-')[:5])] = [result]

# Mark each video with split
task_splits = dict()
for split_name, split_tasks in splits_dict.items():
    for t in split_tasks:
        task_splits[t["task"]] = split_name

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        os.makedirs(dst, exist_ok=True, mode=0o777)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

counter = 1
for trial, trial_dicts in trials.items():
    
    print(counter, '/', len(trials), ' - ', trial)
    # generate original path based on which split the video belongs to in the alfred data
    splits = trial.split('trial')
    for original_split in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]:
        original_path = alfred_data_dir + original_split + "/" + \
            splits[0][:len(splits[0])-1] + '/trial' + splits[1]
        if os.path.exists(alfred_data_dir + original_split + "/" + \
            splits[0][:len(splits[0])-1] + '/trial' + splits[1]):
            break
        
    trial_path = splits[0][:len(splits[0])-1] + '/trial' + splits[1]
    new_split = task_splits[trial_path]
    if os.path.exists(annotations_dir + "/" + new_split + "/" + trial_path):
        continue
    # create folder for trial based on our splits
    new_path = annotations_dir + "/" + new_split + "/" + trial_path
    copytree(original_path, new_path)
    
    # make a copy of the original alfred annotations
    shutil.copy(new_path +"/"+ "traj_data.json", new_path +"/"+ "alfred_traj_data.json")
    
    # edit the original alfred traj_data.json to include transcriptions of our annotations instead of Alfred's
    with open(new_path +"/"+ "traj_data.json", "r") as infile:
        traj = json.load(infile)
    
    our_anns = []
    for trial_dict in trial_dicts:
        trial_ann = {'assignment_id': "_".join(trial_dict["task"].split("-")[-2:]),
                     'high_descs': trial_dict["step_transcriptions"],
                     'task_desc': trial_dict["summary_transcription"],
                     'task_intent': trial_dict["intention_transcription"],
                     'worker_id':trial_dict["worker_id"]}
        our_anns.append(trial_ann)
    
    traj["turk_annotations"]["anns"] = our_anns
    with open(os.open(new_path +"/"+ "traj_data.json", os.O_CREAT | os.O_WRONLY, 0o777), "w") as outfile:
        json.dump(traj, outfile, indent=4)
        counter += 1
        
print('Done')
