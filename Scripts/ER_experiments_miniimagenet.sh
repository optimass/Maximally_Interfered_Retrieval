#!/bin/bash
mem=100
n_iters=3
runs=15

#Permuted MNIST
python er_main.py --method mir_replay --lr 0.1 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --subsample 25  --disc_iters $n_iters --mem_size $mem  --suffix 'ER_MIR'
python er_main.py --method rand_replay --lr 0.1 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --disc_iters $n_iters --mem_size $mem --suffix 'ER'




