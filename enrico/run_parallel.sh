#!/bin/bash

start_seed=0 # start from 10 
num_seeds=4

for ((i=0; i<num_seeds; i++))
do
    seed=$((start_seed + i))
    python3 run_training.py --config enrico.yaml --seed $seed &
done
wait

