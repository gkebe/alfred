import json
import random
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--alfred_data_dir', type=str, help='json_feat_2.1.0 directory from the original Alfred data', required=True)
parser.add_argument('--in_file', type=str, help='a jsonl file of completed task annotations')
parser.add_argument('--train_percent', type=float, help='A number between 0.0 and 1.0')
parser.add_argument('--seed', type=float, help='A float to seed the shuffle. Default is 0.5',
                    default=0.5)
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

for original_split in ["test_seen", "test_unseen"]:
    for trial in os.listdir(os.path.join(args.alfred_data_dir, original_split)):
        with open(os.path.join(alfred_data_dir, original_split, trial, "traj_data.json"), "r") as f:
            traj_data = json.loads(f.read())
            scene = traj_data["scene"]["floor_plan"]
            task_dict[os.path.join(alfred_data_dir, original_split, trial)] = scene