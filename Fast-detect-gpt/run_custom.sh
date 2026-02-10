#!/usr/bin/env bash
set -e

exp_path=exp_custom
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p "$data_path" "$res_path"

sampling_model=${1:-gpt-j-6B}
scoring_model=${2:-gpt-neo-2.7B}
input_jsonl=${3:-/app/datasets/visual_understanding/sample_001.jsonl}
prefix=${4:-custom}
dataset_name=${5:-custom}

python scripts/build_custom_dataset.py \
  --input_jsonl "$input_jsonl" \
  --output_dir "$data_path" \
  --prefix "$prefix"

for variant in onlyllm rag; do
  echo "$(date)", Running Fast-DetectGPT on "${prefix}_${variant}"
  python scripts/fast_detect_gpt.py \
    --sampling_model_name "$sampling_model" \
    --scoring_model_name "$scoring_model" \
    --discrepancy_analytic \
    --dataset "$dataset_name" \
    --dataset_file "$data_path/${prefix}_${variant}" \
    --output_file "$res_path/${prefix}_${variant}.${sampling_model}_${scoring_model}"
done
