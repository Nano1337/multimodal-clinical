#!/bin/bash

start_seed=30 # start from 10 
num_seeds=20

for ((i=0; i<num_seeds; i++))
do
    seed=$((start_seed + i))
    python3 run_training.py --config mimic.yaml --seed $seed &
done
wait

