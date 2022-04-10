#!/bin/bash

data_dir=$(dirname $0)
python ${data_dir}/generate_splits.py --in_file ${data_dir}/shared/coded_transcriptions.jsonl --tests_percent 0.15\
 --valid_percent 0.15 --seed 71 --out_file ${data_dir}/shared/${1}

python ${data_dir}/move_files.py --alfred_data_dir ${data_dir}/shared/json_feat_2.1.0/ --output_dir ${data_dir}/shared/${2}\
 --transcriptions_json ${data_dir}/shared/coded_transcriptions.jsonl --splits_json ${data_dir}/shared/${1}