import json
import random
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, help='a jsonl file of completed task annotations')
parser.add_argument('--tests_percent', type=float, help='A number between 0.0 and 1.0')
parser.add_argument('--train_percent', type=float, help='A number between 0.0 and 1.0')
parser.add_argument('--valid_percent', type=float, help='A number between 0.0 and 1.0')
parser.add_argument('--seed', type=float, help='A float to seed the shuffle. Default is 0.5',
                    default=0.5)
parser.add_argument('--out_file', type=str, help='File to write output json to')
args = parser.parse_args()


if args.tests_percent + args.train_percent + args.valid_percent > 1.0:
    print("Invalid split breakdowns, total is more than 100%")
    sys.exit(1)


parent_dict = {}
task_list = []
task_dict = {}

#Open input file and reformat data into {idx, task}
with open(args.in_file) as src_file:
    for line in src_file:
        tmp_dict = json.loads(line)
        new_dict = {}
        new_dict['repeat_idx'] = 0
        #Remove metadata
        fmtd_task = tmp_dict['task'][:tmp_dict['task'][:(tmp_dict['task'].rfind('-'))].rfind('-')]
        tmp = fmtd_task.replace("_trial", "/trial")
        fmtd_task = tmp
        new_dict['task'] = fmtd_task
        if fmtd_task not in task_list:
            task_list.append(fmtd_task)
            task_dict.update({fmtd_task : []})
        while new_dict in task_dict[fmtd_task]:
            new_dict['repeat_idx'] += 1
        task_dict[fmtd_task].append(new_dict)
src_file.close()

#Shuffle list in place using seed
random.seed(args.seed)
random.shuffle(task_list)

#Generate splits based on input percents
length = len(task_list)
breakOne = int(length * float(args.tests_percent))
breakTwo = int(length * (float(args.tests_percent) + float(args.train_percent)))
test_splits_a = task_list[0:breakOne]
train_splits_a = task_list[breakOne:breakTwo]
validate_splits_a = task_list[breakTwo:]

test_splits = []
train_splits = []
valid_splits = []

for task in test_splits_a:
    for item in task_dict[task]:
        test_splits.append(item)
for task in train_splits_a:
    for item in task_dict[task]:
        train_splits.append(item)
for task in validate_splits_a:
    for item in task_dict[task]:
        valid_splits.append(item)

#Reformat test_splits to remove task name data
#for entry in test_splits:
#    new_task = entry['task'][entry['task'].index('/') + 1 : ]
#    entry['task'] = new_task

parent_dict["tests"] = test_splits
parent_dict["train"] = train_splits
parent_dict["valid"] = valid_splits


with open(args.out_file, "w") as target:
    json.dump(parent_dict, target, indent = 4)

os.chmod(args.out_file, 0o777)
target.close()
