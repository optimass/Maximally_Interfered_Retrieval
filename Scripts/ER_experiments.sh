#!/bin/bash
mems=(100 50 20)
n_iters=1

runs=15
#CIFAR-10
for mem in "${mems[@]}"
    do
        python er_main.py --method mir_replay --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --n_runs $runs --subsample 50  --disc_iters $n_iters --mem_size $mem  --suffix 'ER_MIR'
        python er_main.py --method rand_replay --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --n_runs $runs --disc_iters $n_iters --mem_size $mem --suffix 'ER'
    done

runs=20
#MNIST Split
for mem in "${mems[@]}"
    do
        python er_main.py --method mir_replay --lr 0.1 --samples_per_task 1000 --dataset split_mnist  --subsample 50  --n_runs $runs --disc_iters $n_iters --mem_size $mem --compare_to_old_logits --suffix 'ER_MIR'
        python er_main.py --method rand_replay --lr 0.1 --samples_per_task 1000 --dataset split_mnist --n_runs $runs --disc_iters $n_iters --mem_size $mem --suffix 'ER'
    done


#Permuted MNIST
for mem in "${mems[@]}"
    do
        python er_main.py --method mir_replay --lr 0.05 --samples_per_task 1000 --dataset permuted_mnist --subsample 50 --n_runs $runs --disc_iters $n_iters --mem_size $mem  --compare_to_old_logits --suffix 'ER_MIR'
        python er_main.py --method rand_replay --lr 0.05 --samples_per_task 1000 --dataset permuted_mnist   --n_runs $runs --disc_iters $n_iters --mem_size $mem --suffix 'ER'
     done




