import numpy as np
import pdb
import os
import time
import sys
from numpy.random import choice

debug_mode = 0

sys.path.append(os.getcwd())

runs = 1000
run_counter = 0

n_runs = 3

# fixed
dataset = 'split_mnist'
result_dir = 'split-mnist_hparam_search'
log = 'online'

if debug_mode:
    result_dir = 'temp'

# fixed hparams:
samples = 1000
batch_size = 10
reuse_sample = 1
cls_hiddens = 400

while run_counter < runs:

    method = choice(['rand_gen', 'mir_gen'])

    gen_method = choice(['rand_gen', 'mir_gen'])

    dropout = choice([0., 0.1, 0.2, 0.3, 0.4])

    lr = choice([1e-3, 1e-3, 5e-4])

    max_beta = choice([0.2, 0.5, 1.0, 1.0 ])

    warmup = choice([1, 250, 500, 1000])

    n_mem = choice([1, 2, 4, 10])

    mem_coeff = choice([1, 2, 3, 5])

    #---- Architecture ----#
    z_size = choice([50, 100])
    gen_depth = choice([1,2,3])
    gen_hiddens = choice([128, 256])

    n_iter = choice([2, 5, 10, 15, 20])

    mir_iters = -1
    mir_init_prior = -1


    coeff_values = [0., 0., 0.1, 1., 2.0]

    if gen_method == 'mir_gen':

        mir_iters = choice([2, 3, 5, 10])
        mir_init_prior = choice([0, 1])

        gen_kl_coeff  = choice(coeff_values)
        gen_rec_coeff  = choice(coeff_values)
        gen_ent_coeff  = choice(coeff_values)
        gen_div_coeff = choice(coeff_values)
        gen_shell_coeff = choice(coeff_values)

    else:

        gen_kl_coeff = -1
        gen_rec_coeff = -1
        gen_ent_coeff = -1
        gen_div_coeff = -1
        gen_shell_coeff = -1

    if method == 'mir_gen':

        mir_iters = choice([2, 3, 5, 10])
        mir_init_prior = choice([0, 1])

        cls_xent_coeff  = choice(coeff_values)
        cls_ent_coeff  = choice(coeff_values)
        cls_div_coeff  = choice(coeff_values)
        cls_shell_coeff = choice(coeff_values)

    else:

        cls_xent_coeff = -1
        cls_ent_coeff = -1
        cls_div_coeff = -1
        cls_shell_coeff = -1

    #-------------------------------------------------------

    cwd = os.getcwd()
    command = "python3 gen_main.py \
        --dataset %(dataset)s \
        --n_runs %(n_runs)s \
        --log %(log)s \
        --result_dir %(result_dir)s \
        --method %(method)s \
        --gen_method %(gen_method)s \
        --samples_per_task %(samples)s \
        --z_size %(z_size)s \
        --gen_depth %(gen_depth)s \
        --gen_hiddens %(gen_hiddens)s \
        --cls_hiddens %(cls_hiddens)s \
        --batch_size %(batch_size)s \
        --dropout %(dropout)s \
        --gen_iters %(n_iter)s \
        --cls_iters %(n_iter)s \
        --max_beta %(max_beta)s \
        --warmup %(warmup)s \
        --lr %(lr)s \
        --n_mem %(n_mem)s \
        --mem_coeff %(mem_coeff)s \
        --reuse_sample %(reuse_sample)s \
        --mir_iters %(mir_iters)s \
        --mir_init_prior %(mir_init_prior)s \
        --gen_kl_coeff %(gen_kl_coeff)s \
        --gen_rec_coeff %(gen_rec_coeff)s \
        --gen_ent_coeff %(gen_ent_coeff)s \
        --gen_div_coeff %(gen_div_coeff)s \
        --gen_shell_coeff %(gen_shell_coeff)s \
        --cls_xent_coeff %(cls_xent_coeff)s \
        --cls_ent_coeff %(cls_ent_coeff)s \
        --cls_div_coeff %(cls_div_coeff)s \
        --cls_shell_coeff %(cls_shell_coeff)s \
        " % locals()

    if debug_mode:
        command += ' -u'

    print(command)

    os.system(command)
    time.sleep(2)
    run_counter += 1

