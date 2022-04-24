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

#Open input file and reformat data into {idx, task}
with open(args.in_file) as src_file:
    for line in src_file:
        tmp_dict = json.loads(line)
        fmtd_task = tmp_dict['task'][:tmp_dict['task'][:(tmp_dict['task'].rfind('-'))].rfind('-')]
        tmp = fmtd_task.replace("_trial", "/trial")
        for original_split in ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]:
            if os.path.exists(os.path.join(alfred_data_dir, original_split, fmtd_task, "traj_data.json")):
                with open(os.path.join(alfred_data_dir, original_split, fmtd_task, "traj_data.json")) as f:
                    traj_data = json.loads(f)
        scene = traj_data["scene"]
        print(tmp_dict['task'])
        print(scene)