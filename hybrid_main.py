import os
import time
import wandb
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from collections import OrderedDict as OD
from collections import defaultdict as DD

from data   import *
from mir    import *
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, CVAE
from utils  import get_logger, get_temp_logger, logging_per_task

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='Results',
    help='directory where we save results and samples')
parser.add_argument('--dataset', type=str, choices=['split_mnist', 'permuted_mnist', 'split_cifar10', 'split_cifar100', 'miniimagenet'], default = 'split_cifar10')
parser.add_argument('--n_tasks', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--use_conv', action='store_true')
parser.add_argument('--samples_per_task', type=int, default=-1, help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=600, help='controls buffer size') # mem_size in the tf repo.
parser.add_argument('--n_runs', type=int, default=1, help='number of runs to average performance')
parser.add_argument('--n_iters', type=int, default=1, help='training iterations on incoming batch')
parser.add_argument('--rehearsal', type=int, default=1, help='whether to replay previous data')
parser.add_argument('--max_loss', action='store_true', help='pick samples that maximize kl before after minibatch update')
parser.add_argument('--full_ab', type=int, default=0)

# latent buffer
parser.add_argument('--store_latents', type=int, default=1)
parser.add_argument('--gen_rehearsal', type=int, default=1)
parser.add_argument('--update_buffer_hid', type=int, default=1)

# interfered retrieval
parser.add_argument('--max_loss_budget', type=int, default=1)
parser.add_argument('--max_loss_grad_steps', type=int, default=5)
parser.add_argument('--both_entropy', type=int, default=1)
parser.add_argument('--reuse_samples', type=int, default=0)

# coefficients
parser.add_argument('--kl_coef', type=float, default=200)
parser.add_argument('--ent_coef', type=float, default=1e-2)
parser.add_argument('--euc_coef', type=float, default=0)
parser.add_argument('--mem_strength', type=float, default=0.8)

# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online'],
    help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='mir',
    help='name of the WandB project')

args = parser.parse_args()

# Obligatory overhead
# -----------------------------------------------------------------------------------------
if not os.path.exists(args.result_dir): os.makedirs(args.result_dir, exist_ok=True)

# fixed for now
args.device = 'cuda:0'
args.gen = False
args.meta_buffer = 0
args.newer = 2
args.output_loss = 'mse'
args.cuda = True

if args.log != 'off':
    wandb.init(args.wandb_project)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['gen_loss', 'cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)

# Functions
# -----------------------------------------------------------------------------------------
mean_fn    = lambda x : sum(x) / len(x)


# Train the model
# -----------------------------------------------------------------------------------------

for run in range(args.n_runs):
    # reproducibility is da best
    np.random.seed(run)
    torch.manual_seed(run)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # fetch data
    data = locate('data.get_%s' % args.dataset)(args)

    # make dataloaders
    train_loader, val_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False, False])]


    model = ResNet18(args.n_classes, nf=20, input_size=args.input_size).to(args.device)
    opt   = torch.optim.SGD(model.parameters(), lr=0.1)

    gen     = CVAE(20, args).cuda() # this is actually an autoencoder
    opt_gen = torch.optim.Adam(gen.parameters())

    # build buffer
    if args.store_latents:
        buffer = Buffer(args, input_size = (20*4*4,))
    else:
        buffer = Buffer(args)

    buffer.min_per_class = 0
    print('multiple heads ', args.multiple_heads)

    if run == 0:
        print("number of classifier parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        print("number of generator parameters: ", sum([np.prod(p.size()) for p in gen.parameters()]))
        print("buffer parameters:              ", np.prod(buffer.bx.size()))

    prev_gen, prev_model = None, None
    best_acc_yet = {'valid':np.zeros((args.n_tasks)), 'test':np.zeros((args.n_tasks))}

    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0

        model = model.train()

        # iterate over samples from task
        for epoch in range(args.n_epochs):

            # create logging containers
            LOG_temp = get_temp_logger(None, ['gen_loss', 'cls_loss', 'acc'])

            for i, (data, target) in enumerate(tr_loader):
                if sample_amt > args.samples_per_task > 0: break
                sample_amt += data.size(0)

                data, target = data.to(args.device), target.to(args.device)

                for it in range(args.n_iters if epoch == args.n_epochs - 1 else 1):
                    _, track_idx = buffer.split(0)
                    input_x, input_y = data, target
                    task_ids  = torch.zeros_like(input_y).fill_(task)

                    if task > 0 and track_idx.nelement() > 0 and args.rehearsal:

                        # ----------------- #
                        if args.max_loss and epoch == args.n_epochs - 1 and (it == 0 or not args.reuse_samples):
                            mem_x, mem_y, b_task_ids = retrieve_hybrid(args, model, gen, prev_model, prev_gen, input_x, input_y, buffer, task)
                        # ----------------- #
                        else:
                            mem_x, mem_y, b_task_ids = buffer.sample(args.buffer_batch_size , exclude_task = task)

                        if args.store_latents:
                            with torch.no_grad():
                                mem_x = prev_gen.decode(mem_x)

                        input_x = torch.cat((input_x, mem_x))
                        input_y = torch.cat((input_y, mem_y))
                        task_ids = torch.cat((task_ids, b_task_ids))

                    # train generator
                    if True : #epoch < args.n_epochs - 1:
                        opt_gen.zero_grad()
                        if args.gen_rehearsal:
                            x_recon, hid = gen(input_x)
                            gen_loss = F.mse_loss(input_x, x_recon)
                        else:
                            x_recon, hid = gen(data)
                            gen_loss = F.mse_loss(data, x_recon)

                        gen_loss.backward()
                        opt_gen.step()

                    # train classifer online (1 pass through data)
                    if epoch == args.n_epochs - 1:

                        # we never want the classifier to see real data
                        if task > 0 and (not args.full_ab):
                            # input_x = (current_batch, rehearse) --> (recon_current, rehearse)
                            input_x = torch.cat((x_recon[:data.size(0)].detach(), input_x[data.size(0):]))

                        logits = model(input_x)

                        if args.multiple_heads:
                            mask = torch.zeros_like(logits)
                            mask.scatter_(1, tr_loader.dataset.task_ids[task_ids], 1)
                            logits  = logits.masked_fill(mask == 0, -1e9)

                        if task > 0:
                            loss = F.cross_entropy(logits, input_y, reduction='none')
                            loss_t, loss_re = loss[:data.size(0)], loss[data.size(0):]
                            loss = args.mem_strength * loss_re.sum() + (1 - args.mem_strength) * loss_t.sum()
                            loss = loss / logits.size(0)
                        else:
                            loss = F.cross_entropy(logits, input_y)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        pred = logits.argmax(dim=1, keepdim=True)

                        LOG_temp['acc'] += [pred.eq(input_y.view_as(pred)).sum().item() / pred.size(0)]
                        LOG_temp['cls_loss'] += [loss.item()]

                    LOG_temp['gen_loss'] += [gen_loss.item()]

                # add to buffer only during last epoch of task to get the best reconstructions
                if epoch == args.n_epochs - 1:

                    # buffer.add_reservoir(data, target, None, task)
                    split = data.size(0)

                    if args.store_latents:
                        buffer.add_reservoir(hid[:split], target, None, task)
                    else:
                        buffer.add_reservoir(x_recon[:split], target, None, task)

            LOG_temp.print_('train', task)

            # ------------------------ eval ------------------------ #

            # We need to update the buffer representations
            with torch.no_grad():

                if prev_gen is not None and args.store_latents and args.update_buffer_hid:

                    # sample from buffer
                    for i in range(buffer.bx.data.size(0) // 64):
                        indices = range(i * 64, min((i+1) * 64, buffer.x.data.size(0)))
                        enc_ind = buffer.bx.data[indices]

                        if enc_ind.size(0) > 0:
                            # fetch representations from embeddings
                            x_re = prev_gen.decode(enc_ind)
                            hid  = gen(x_re)[-1]

                            buffer.bx[indices] = hid

                # udpate previous model
                prev_gen = deepcopy(gen)
                prev_model = deepcopy(model)

                if epoch == args.n_epochs - 1:
                    model = model.eval()

                    eval_loaders = [('valid', val_loader), ('test', test_loader)]

                    for mode, loader_ in eval_loaders:
                        current_acc = np.zeros((args.n_tasks))

                        for task_t, te_loader in enumerate(loader_):
                            if task_t > task: break
                            LOG_temp = get_temp_logger(None, ['gen_loss', 'cls_loss', 'acc'])

                            # iterate over samples from task
                            for i, (data, target) in enumerate(te_loader):
                                data, target = data.to(args.device), target.to(args.device)

                                # since the model never sees real images, we autoencode the data
                                x_recon = gen(data)[0]

                                logits = model(x_recon)

                                # DOING ABLATION
                                # logits = model(data)

                                if args.multiple_heads:
                                    logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

                                loss = F.cross_entropy(logits, target)
                                pred = logits.argmax(dim=1, keepdim=True)

                                LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
                                LOG_temp['cls_loss'] += [loss.item()]
                                LOG_temp['gen_loss'] += [F.mse_loss(x_recon, data).item()]

                            current_acc[task_t]  = mean_fn(LOG_temp['acc'])

                            logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                                     np.round(np.mean(LOG_temp['acc']),2))
                            logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                                     np.round(np.mean(LOG_temp['cls_loss']),2))
                            logging_per_task(wandb, LOG, run, mode, 'gen_loss', task, task_t,
                                     np.round(np.mean(LOG_temp['gen_loss']),2))

                        print('\n{}:'.format(mode))
                        print(LOG[run][mode]['acc'])

                        # store the best accuracy seen so far to all the tasks
                        best_acc_yet[mode] = np.maximum(best_acc_yet[mode], current_acc)


    # final run results
    print('--------------------------------------')
    print('Run #{} Final Results'.format(run))
    print('--------------------------------------')
    for mode in ['valid','test']:
        final_accs = LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_acc', task,
            value=np.round(np.mean(final_accs),2))
        best_acc = np.max(LOG[run][mode]['acc'], 1)
        final_forgets = best_acc - LOG[run][mode]['acc'][:,task]
        logging_per_task(wandb, LOG, run, mode, 'final_forget', task,
            value=np.round(np.mean(final_forgets),2))

        print('\n{}:'.format(mode))
        print('final accuracy: {}'.format(final_accs))
        print('average: {}'.format(LOG[run][mode]['final_acc']))
        print('final forgetting: {}'.format(final_forgets))
        print('average: {}\n'.format(LOG[run][mode]['final_forget']))

# final results
print('--------------------------------------')
print('--------------------------------------')
print('FINAL Results')
print('--------------------------------------')
print('--------------------------------------')
for mode in ['valid','test']:

    final_accs = [LOG[x][mode]['final_acc'] for x in range(args.n_runs)]
    final_acc_avg = np.mean(final_accs)
    final_acc_se = np.std(final_accs) / np.sqrt(args.n_runs)
    final_forgets = [LOG[x][mode]['final_forget'] for x in range(args.n_runs)]
    final_forget_avg = np.mean(final_forgets)
    final_forget_se = np.std(final_forgets) / np.sqrt(args.n_runs)

    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))

    if wandb is not None:
        wandb.log({mode+'final_acc_avg':final_acc_avg})
        wandb.log({mode+'final_acc_se':final_acc_se})
        wandb.log({mode+'final_forget_avg':final_forget_avg})
        wandb.log({mode+'final_forget_se':final_forget_se})

# save log file in result dir
delattr(LOG, 'print_')
with open(os.path.join(args.result_dir, 'log'), 'wb') as handle:
        pickle.dump(LOG, handle, protocol=pickle.HIGHEST_PROTOCOL)

time.sleep(2)
