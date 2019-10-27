'''
This script reproduces the results related to gen-mir
runs 2 datasets {split-mnist and permuted-mnist}:
    runs 4 models:
        - rand_gen: baseline (generative replay)
        - gen_mir_gen: mir_gen applied to generator
        - cls_mir_gen: mir_gen applied to classifier
        - mir_gen: mir_gen applied to both classifier and generator

'''
import numpy as np
import pdb
import os
import time
import sys
from numpy.random import choice

debug_mode = 1

sys.path.append(os.getcwd())

n_runs = 20

'''
pseudo-code:

for split_mnist:
    for acc:
        run all 4 configurations (because ablation)
    for elbo:
        run baseline and mir_gen

for permuted_mnist:
    for acc:
        run baseline and cls_mir_gen
    for elbo:
        run baseline and gen_mir_gen
'''

DATASETS = ['split_mnist', 'permuted_mnist']
METHODS = ['mir_gen', 'rand_gen']
GEN_METHODS = ['mir_gen', 'rand_gen']
METRICS = ['acc']
log = 'online'


# fixed hparams
samples = 1000
cls_hiddens = 400
batch_size = 10
reuse_sample = 1

# search for but always the same
mir_init_prior = 1

for dataset in DATASETS:

    result_dir = 'final_{}'.format(dataset)
    if debug_mode:
        result_dir = 'temp'

    for metric in METRICS:

        for method in METHODS:

            for gen_method in GEN_METHODS:

                # by default, turn off MIR stuff
                mir_iters = 0           ; cls_xent_coeff = 0        ; cls_ent_coeff = 0
                cls_div_coeff = 0       ; cls_shell_coeff = 0       ; gen_kl_coeff = 0
                gen_rec_coeff = 0       ; gen_ent_coeff = 0         ; gen_div_coeff = 0
                gen_shell_coeff = 0


                #--------------------------------------------#
                ''' This is the generative replay baseline '''
                if method=='rand_gen' and gen_method == 'rand_gen':

                    if dataset == 'split_mnist':

                        if metric == 'acc':
                            '''
                            valid acc (hparam search): 80.0%
                            test acc:    79.3% +/- 0.6%
                            test forget: 19.5% +/- 0.8%
                            '''

                            name = 'best_baseline'  ; dropout = 0.     ; lr = 0.01
                            max_beta = 1            ; warmup = 1000    ; n_mem = 4
                            mem_coeff = 3           ; z_size = 50      ; gen_depth = 2
                            gen_hiddens = 128       ; n_iter = 15

                        if metric == 'elbo':
                            '''
                            valid elbo (hparam search): 107.9
                            test elbo: 107.2 +/- 0.2
                            '''

                            name = 'best_vae_baseline'   ; dropout = 0.    ; lr = 0.01
                            max_beta = 0.2               ; warmup = 500    ; n_mem = 10
                            mem_coeff = 2                ; z_size = 50     ; gen_depth = 1
                            gen_hiddens = 128            ; n_iter = 15

                    if dataset == 'permuted_mnist':

                        if metric == 'acc':
                            '''
                            valid acc (hparam search): 79.0%
                            test acc: 79.7 +/- 0.1%
                            test forget: 5.8% +/- 0.2%
                            '''

                            name = 'best_baseline'      ; dropout = 0.      ; lr = 0.1
                            max_beta = 0.2              ; warmup = 250      ; n_mem = 4
                            mem_coeff = 2               ; z_size = 50       ; gen_depth = 1
                            gen_hiddens = 128           ; n_iter = 15

                        if metric == 'elbo':
                            '''
                            valid elbo (hparam search): 197.8
                            test elbo: 196.7 +/- 0.7
                            '''

                            name = 'best_vae_baseline'  ; dropout = 0.      ; lr = 0.1
                            max_beta = 0.5              ; warmup = 1000     ; n_mem = 10
                            mem_coeff = 5               ; z_size = 100      ; gen_depth = 2
                            gen_hiddens = 256           ; n_iter = 20


                #---------------------------------------------#
                '''This is MIR only applied to the generator'''
                if method=='rand_gen' and gen_method=='mir_gen':

                    if dataset == 'split_mnist':

                        if metric == 'acc':
                            '''
                            valid (hparam search): 81.3%
                            test acc:    81.4% +/- 0.5%
                            test forget: 16.5% +/- 0.6%
                            '''

                            name = 'best_gen_mir_gen'   ; dropout = 0.      ; lr = 0.01
                            max_beta = 1                ; warmup = 500      ; n_mem = 4
                            mem_coeff = 3               ; z_size = 50       ; gen_depth = 2
                            gen_hiddens = 128           ; n_iter = 10

                            mir_iters = 2

                            #coeff
                            gen_rec_coeff = 0.1
                            gen_kl_coeff = 0.1

                        elif metric=='elbo':
                            pass


                    if dataset == 'permuted_mnist':

                        if metric == 'acc':
                            pass

                        if metric == 'elbo':
                            '''
                            valid elbo: (hparam search): 192.7
                            test elbo: 193.7 +/- 1.0
                            '''

                            name='best_vae_gen_mir_gen'     ; dropout = 0.      ; lr = 0.1
                            max_beta = 0.2                  ; warmup = 1        ; n_mem = 10
                            mem_coeff = 3                   ; z_size = 50       ; gen_depth = 2
                            gen_hiddens = 128               ; n_iter = 15

                            mir_iters = 10

                            #coeff
                            gen_div_coeff = 2
                            gen_shell_coeff = 2


                #---------------------------------------------#
                '''This is MIR only applied to the classifier'''
                if method=='mir_gen' and gen_method=='rand_gen':

                    if dataset == 'split_mnist':

                        if metric == 'acc':
                            '''
                            valid (hparam search): 82.3%
                            test acc:    82.9% +/- 0.3%
                            test forget: 13.6% +/- 0.4%
                            '''

                            name='best_cls_mir_gen'     ; dropout = 0.      ; lr = 0.01
                            max_beta = 0.2              ; warmup = 1000     ; n_mem = 10
                            mem_coeff = 5               ; z_size = 50       ; gen_depth = 1
                            gen_hiddens = 256           ; n_iter = 10

                            mir_iters = 3

                            #coeff
                            cls_xent_coeff = 0.1
                            cls_ent_coeff = 1
                            cls_div_coeff = 2

                        elif metric=='elbo':
                            pass

                    if dataset == 'permuted_mnist':

                        if metric == 'acc':
                            '''
                            valid acc (hparam search): 79.3%
                            test acc: 80.4% +/- 0.2%
                            test forget: 4.8% +/- 0.2%
                            '''

                            name='best_cls_mir_gen'     ; dropout = 0.      ; lr = 0.1
                            max_beta = 0.5              ; warmup = 1        ; n_mem = 10
                            mem_coeff = 3               ; z_size = 50       ; gen_depth = 1
                            gen_hiddens = 128           ; n_iter = 20

                            mir_iters = 2

                            #coeff
                            cls_xent_coeff = 0.1
                            cls_ent_coeff = 1
                            cls_div_coeff = 1
                            cls_shell_coeff = 2

                        if metrics == 'elbo':
                            pass


                #---------------------------------------------#
                if method=='mir_gen' and gen_method=='mir_gen':
                    '''This is MIR applied to classifier and generator'''

                    if dataset == 'split_mnist':

                        if metric == 'acc':
                            '''
                            valid acc (hparam search): 83%
                            test acc: 82.1% +/- 0.3%
                            test forget: 17.0% +/- 0.4%
                            '''

                            name='best_mir_gen'     ; dropout = 0.4     ; lr = 0.01
                            max_beta = 1            ; warmup = 1000     ; n_mem = 10
                            mem_coeff = 3           ; z_size = 50       ; gen_depth = 1
                            gen_hiddens = 128       ; n_iter = 15

                            mir_iters = 10

                            #coeff
                            cls_xent_coeff = 0.1
                            cls_ent_coeff = 1
                            cls_div_coeff = 1
                            gen_kl_coeff = 0.1
                            gen_ent_coeff = 1
                            gen_div_coeff = 1

                        elif metric == 'elbo':
                            '''
                            valid elbo (hparam search): 102.9
                            test elbo: 102.5 +/- 0.2
                            '''

                            name='best_vae_mir_gen'     ; dropout = 0.      ; lr = 0.1
                            max_beta = 0.2              ; warmup = 500      ; n_mem = 10
                            mem_coeff = 5               ; z_size = 50       ; gen_depth = 1
                            gen_hiddens = 256           ; n_iter = 20

                            mir_iters = 2

                            #coeff
                            cls_xent_coeff = 1
                            cls_ent_coeff = 1
                            cls_div_coeff = 0.1
                            gen_ent_coeff = 2
                            gen_div_coeff = 0.1
                            gen_shell_coeff = 1


                    if dataset == 'permuted_mnist':
                        pass



                #-------------------------------------------------------

                cwd = os.getcwd()
                command = "python3 gen_main.py \
                    --dataset %(dataset)s \
                    --n_runs %(n_runs)s \
                    --log %(log)s \
                    --result_dir %(result_dir)s \
                    --method %(method)s \
                    --gen_method %(gen_method)s \
                    --name %(name)s \
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
                    pass
                    #command += ' -u'

                print(command)

                os.system(command)
                time.sleep(2)

