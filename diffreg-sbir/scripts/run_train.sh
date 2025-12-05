#!/bin/bash
# Example usage:
# bash scripts/run_train.sh configs/default.yaml
CONFIG=$1
python train.py --config $CONFIG