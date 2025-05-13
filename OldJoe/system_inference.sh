#!/bin/bash

SPLIT="test_2025"
BASE_PATH="/home/ubuntu/data/"

# Run fact-checker
python -m veritas.scripts.run hydra.run.dir="results/${SPLIT}" claim_fpath="${BASE_PATH}/${SPLIT}.json" || exit 1
# Convert JSON to CSV
python prepare_leaderboard_submission.py --filename "results/${SPLIT}/results.json" --save_file "results/${SPLIT}/results.csv" || exit 1
# Evaluate results from JSON
python averitec_evaluate.py --prediction_file "results/${SPLIT}/results.json"  --label_file "/home/ubuntu/AVeriTeC/data/${SPLIT}.json" 
