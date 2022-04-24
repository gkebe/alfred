import json
import random
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--alfred_data_dir', type=str, help='json_feat_2.1.0 directory from the original Alfred data', required=True)
parser.add_argument('--out_file', type=str, help='File to write output json to')
args = parser.parse_args()

parent_dict = {}
task_list = []
task_dict = {}

alfred_data_dir = args.alfred_data_dir
#Open input file and reformat data into {idx, task}
for original_split in ["train", "valid_seen", "valid_unseen"]:
    for task in os.listdir(os.path.join(args.alfred_data_dir, original_split)):
        for trial in os.listdir(os.path.join(args.alfred_data_dir, original_split, task)):
            with open(os.path.join(alfred_data_dir, original_split, task, trial, "traj_data.json"), "r") as f:
                traj_data = json.loads(f.read())
                scene = traj_data["scene"]["floor_plan"]
                task_dict[os.path.join(alfred_data_dir, original_split, task, trial)] = scene

for original_split in ["tests_seen", "tests_unseen"]:
    for trial in os.listdir(os.path.join(args.alfred_data_dir, original_split)):
        with open(os.path.join(alfred_data_dir, original_split, trial, "traj_data.json"), "r") as f:
            traj_data = json.loads(f.read())
            scene = traj_data["scene"]["floor_plan"]
            task_dict[os.path.join(alfred_data_dir, original_split, trial)] = scene

with open(args.out_file, "w") as target:
    json.dump(task_dict, target, indent = 4)