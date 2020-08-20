import argparse
import torch.nn.functional as F
import numpy as np


from data   import *
from mir    import *
from utils  import get_logger, get_temp_logger, logging_per_task
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, MLP

# Arguments
# -----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='Results',
    help='directory where we save results and samples')
parser.add_argument('-u', '--unit_test', action='store_true',
    help='unit testing mode for fast debugging')
parser.add_argument('-d', '--dataset', type=str, default = 'split_mnist',
    choices=['split_mnist', 'permuted_mnist', 'split_cifar10', 'split_cifar100', 'miniimagenet'])
parser.add_argument('--n_tasks', type=int, default=-1,
    help='total number of tasks. -1 does default amount for the dataset')
parser.add_argument('-r','--reproc', type=int, default=1,
    help='if on, no randomness in numpy and torch')
parser.add_argument('--disc_epochs', type=int, default=1)
parser.add_argument('--disc_iters', type=int, default=1,
    help='number of training iterations for the classifier')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--use_conv', action='store_true')
parser.add_argument('--samples_per_task', type=int, default=-1,
    help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=600, help='controls buffer size')
parser.add_argument('--n_runs', type=int, default=1,
    help='number of runs to average performance')
parser.add_argument('--suffix', type=str, default='',
    help="name for logfile")
parser.add_argument('--subsample', type=int, default=50,
    help="for subsampling in --method=replay, set to 0 to disable")
parser.add_argument('--print_every', type=int, default=500,
    help="print metrics every this iteration")
parser.add_argument('--update_buffer_hid', type=int, default=1,
    help='related to latent buffer')
# logging
parser.add_argument('-l', '--log', type=str, default='off', choices=['off', 'online'],
    help='enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='mir',
    help='name of the WandB project')

#------ MIR -----#
parser.add_argument('-m','--method', type=str, default='no_rehearsal', choices=['no_rehearsal',
    'rand_replay', 'mir_replay'])
parser.add_argument('--compare_to_old_logits', action='store_true',help='uses old logits')
parser.add_argument('--reuse_samples', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args()

# Obligatory overhead
# -----------------------------------------------------------------------------------------

if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
sample_path = os.path.join(args.result_dir,'samples/')
if not os.path.exists(sample_path): os.mkdir(sample_path)
recon_path = os.path.join(args.result_dir,'reconstructions/')
if not os.path.exists(recon_path): os.mkdir(recon_path)
if args.suffix is not '':
    import datetime
    time_stamp = str(datetime.datetime.now().isoformat())
    name_log_txt = args.dataset+'_'+time_stamp + str(np.random.randint(0, 1000)) + args.suffix
    name_log_txt=name_log_txt +'.log'
    with open(name_log_txt, "a") as text_file:
        print(args, file=text_file)
else:
    name_log_txt = None
args.cuda = torch.cuda.is_available()
args.device = 'cuda:0'

# argument validation
overlap = 0

#########################################
# TODO(Get rid of this or move to data.py)
args.ignore_mask = False
args.gen = False
args.newer = 2
#########################################

args.gen_epochs=0
args.output_loss = None

if args.reproc:
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)

# fetch data
data = locate('data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, val_loader, test_loader  = [CLDataLoader(elem, args, train=t) \
        for elem, t in zip(data, [True, False, False])]

if args.log != 'off':
    import wandb
    wandb.init(args.wandb_project)
    wandb.config.update(args)
else:
    wandb = None

# create logging containers
LOG = get_logger(['cls_loss', 'acc'],
        n_runs=args.n_runs, n_tasks=args.n_tasks)

args.mem_size = args.mem_size*args.n_classes #convert from per class to total memory

# Train the model
# -----------------------------------------------------------------------------------------

for run in range(args.n_runs):

    # REPRODUCTIBILITY
    if args.reproc:
        np.random.seed(run)
        torch.manual_seed(run)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLASSIFIER
    if args.use_conv:
        model = ResNet18(args.n_classes, nf=20, input_size=args.input_size)
    else:
        model = MLP(args)
    if args.cuda:
        model = model.to(args.device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    buffer = Buffer(args)
    if run == 0:
        print("number of classifier parameters:",
                sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters: ", np.prod(buffer.bx.size()))

    #----------
    # Task Loop
    for task, tr_loader in enumerate(train_loader):
        sample_amt = 0

        model = model.train()

        #---------------
        # Minibatch Loop
        for i, (data, target) in enumerate(tr_loader):
            if args.unit_test and i > 10: break
            if sample_amt > args.samples_per_task > 0: break
            sample_amt += data.size(0)

            if args.cuda:
                data, target = data.to(args.device), target.to(args.device)


            #------ Train Classifier-------#
            if i==0:
                print('\n--------------------------------------')
                print('Run #{} Task #{} --> Train Classifier'.format(
                    run, task))
                print('--------------------------------------\n')

            #---------------
            # Iteration Loop
            for it in range(args.disc_iters):
                if args.method == 'no_rehearsal':
                    rehearse = False
                else:
                    rehearse = task>0
                model = retrieve_replay_update(args,
                                    model, opt, data, target, buffer, task, tr_loader,rehearse=rehearse)

            buffer.add_reservoir(data, target, None, task)

        # ------------------------ eval ------------------------ #
        model = model.eval()
        eval_loaders = [('valid', val_loader), ('test', test_loader)]

        for mode, loader_ in eval_loaders:
            for task_t, te_loader in enumerate(loader_):
                if task_t > task: break
                LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])

                # iterate over samples from task
                for i, (data, target) in enumerate(te_loader):
                    if args.unit_test and i > 10: break

                    if args.cuda:
                        data, target = data.to(args.device), target.to(args.device)

                    logits = model(data)

                    if args.multiple_heads:
                        logits = logits.masked_fill(te_loader.dataset.mask == 0, -1e9)

                    loss = F.cross_entropy(logits, target)
                    pred = logits.argmax(dim=1, keepdim=True)

                    LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
                    LOG_temp['cls_loss'] += [loss.item()]

                logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                         np.round(np.mean(LOG_temp['acc']),2))
                logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                         np.round(np.mean(LOG_temp['cls_loss']),2))

            print('\n{}:'.format(mode))
            print(LOG[run][mode]['acc'])


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
            value=np.round(np.mean(final_forgets[:-1]),2))

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
    final_acc_se = 2*np.std(final_accs) / np.sqrt(args.n_runs)
    final_forgets = [LOG[x][mode]['final_forget'] for x in range(args.n_runs)]
    final_forget_avg = np.mean(final_forgets)
    final_forget_se = 2*np.std(final_forgets) / np.sqrt(args.n_runs)

    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))

    if name_log_txt is not None:
        with open(name_log_txt, "a") as text_file:
            print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se), file=text_file)
            print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se), file=text_file)

    if wandb is not None:
        wandb.log({mode+'final_acc_avg':final_acc_avg})
        wandb.log({mode+'final_acc_se':final_acc_se})
        wandb.log({mode+'final_forget_avg':final_forget_avg})
        wandb.log({mode+'final_forget_se':final_forget_se})
