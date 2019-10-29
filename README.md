## Code for Online Continual Learning with Maximally Interfered Retrieval (NeurIPS 2019) 

https://arxiv.org/abs/1908.04742

### Experience Replay

For an example of the experience replay mir run 

`python er_main.py --method mir_replay --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --n_runs 1 --subsample 50  --mem_size 50`

and for the baseline:

`python er_main.py --method rand_replay --lr 0.1 --samples_per_task -1 --dataset split_cifar10 --n_runs 1 --mem_size 50`


to run the full set of experiments for MIR-ER:
`bash Scripts\ER_experiments.sh`


