import numpy as np
import math
from copy import deepcopy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_grad_vector, get_future_step_parameters
from VAE.loss import calculate_loss

#----------
# Functions
dist_kl = lambda y, t_s : F.kl_div(F.log_softmax(y, dim=-1), F.softmax(t_s, dim=-1), reduction='mean') * y.size(0)
# this returns -entropy
entropy_fn = lambda x : torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)

cross_entropy = lambda y, t_s : -torch.sum(F.log_softmax(y, dim=-1)*F.softmax(t_s, dim=-1),dim=-1).mean()
mse = torch.nn.MSELoss()

def retrieve_gen_for_cls(args, x, cls, prev_cls, prev_gen):

    grad_vector = get_grad_vector(args, cls.parameters, cls.grad_dims)

    virtual_cls = get_future_step_parameters(cls, grad_vector,
            cls.grad_dims, args.lr)

    _, z_mu, z_var, _, _, _ = prev_gen(x)

    z_new_max = None

    for i in range(args.n_mem):

        with torch.no_grad():

            if args.mir_init_prior:
                z_new = prev_gen.prior.sample((z_mu.shape[0],)).to(args.device)
            else:
                z_new = prev_gen.reparameterize(z_mu, z_var)

        for j in range(args.mir_iters):

            z_new.requires_grad = True

            x_new = prev_gen.decode(z_new)
            y_pre = prev_cls(x_new)
            y_virtual = virtual_cls(x_new)

            # maximise the interference:
            XENT = 0
            if args.cls_xent_coeff>0.:
                XENT = cross_entropy(y_virtual, y_pre)

            # the predictions from the two models should be confident
            ENT = 0
            if args.cls_ent_coeff>0.:
                ENT = cross_entropy(y_pre, y_pre)
            #TODO(should we do the args.curr_entropy thing?)

            # the new found samples samples should be differnt from each others
            DIV = 0
            if args.cls_div_coeff>0.:
                for found_z_i in range(i):
                    DIV += F.mse_loss(
                        z_new,
                        z_new_max[found_z_i * z_new.size(0):found_z_i * z_new.size(0) + z_new.size(0)]
                        ) / (i)

            # (NEW) stay on gaussian shell loss:
            SHELL = 0
            if args.cls_shell_coeff>0.:
                SHELL = mse(torch.norm(z_new, 2, dim=1),
                        torch.ones_like(torch.norm(z_new, 2, dim=1))*np.sqrt(args.z_size))

            gain = args.cls_xent_coeff * XENT + \
                   -args.cls_ent_coeff * ENT + \
                   args.cls_div_coeff * DIV + \
                   -args.cls_shell_coeff * SHELL

            z_g = torch.autograd.grad(gain, z_new)[0]
            z_new = (z_new + 1 * z_g).detach()

        if z_new_max is None:
            z_new_max = z_new.clone()
        else:
            z_new_max = torch.cat([z_new_max, z_new.clone()])

    z_new_max.require_grad = False

    if np.isnan(z_new_max.to('cpu').numpy()).any():
        mir_worked = 0
        mem_x = prev_gen.generate(args.batch_size*args.n_mem).detach()
    else:
        mem_x = prev_gen.decode(z_new_max).detach()
        mir_worked = 1

    mem_y = torch.softmax(prev_cls(mem_x), dim=1).detach()

    return mem_x, mem_y, mir_worked



def retrieve_gen_for_gen(args, x, gen, prev_gen, prev_cls):

    grad_vector = get_grad_vector(args, gen.parameters, gen.grad_dims)

    virtual_gen = get_future_step_parameters(gen, grad_vector, gen.grad_dims, args.lr)

    _, z_mu, z_var, _, _, _ = prev_gen(x)

    z_new_max = None
    for i in range(args.n_mem):

        with torch.no_grad():

            if args.mir_init_prior:
                z_new = prev_gen.prior.sample((z_mu.shape[0],)).to(args.device)
            else:
                z_new = prev_gen.reparameterize(z_mu, z_var)

        for j in range(args.mir_iters):
            z_new.requires_grad = True

            x_new = prev_gen.decode(z_new)


            prev_x_mean, prev_z_mu, prev_z_var, prev_ldj, prev_z0, prev_zk = \
                    prev_gen(x_new)
            _, prev_rec, prev_kl, _ = calculate_loss(prev_x_mean, x_new, prev_z_mu, \
                    prev_z_var, prev_z0, prev_zk, prev_ldj, args, beta=1)

            virtual_x_mean, virtual_z_mu, virtual_z_var, virtual_ldj, virtual_z0, virtual_zk = \
                    virtual_gen(x_new)
            _, virtual_rec, virtual_kl, _ = calculate_loss(virtual_x_mean, x_new, virtual_z_mu, \
                    virtual_z_var, virtual_z0, virtual_zk, virtual_ldj, args, beta=1)

            #TODO(warning, KL can explode)


            # maximise the interference
            KL = 0
            if args.gen_kl_coeff>0.:
                KL = virtual_kl - prev_kl

            REC = 0
            if args.gen_rec_coeff>0.:
                REC = virtual_rec - prev_rec

            # the predictions from the two models should be confident
            ENT = 0
            if args.gen_ent_coeff>0.:
                y_pre = prev_cls(x_new)
                ENT = cross_entropy(y_pre, y_pre)
            #TODO(should we do the args.curr_entropy thing?)

            DIV = 0
            # the new found samples samples should be differnt from each others
            if args.gen_div_coeff>0.:
                for found_z_i in range(i):
                    DIV += F.mse_loss(
                        z_new,
                        z_new_max[found_z_i * z_new.size(0):found_z_i * z_new.size(0) + z_new.size(0)]
                        ) / (i)

            # (NEW) stay on gaussian shell loss:
            SHELL = 0
            if args.gen_shell_coeff>0.:
                SHELL = mse(torch.norm(z_new, 2, dim=1),
                        torch.ones_like(torch.norm(z_new, 2, dim=1))*np.sqrt(args.z_size))

            gain = args.gen_kl_coeff * KL + \
                   args.gen_rec_coeff * REC + \
                   -args.gen_ent_coeff * ENT + \
                   args.gen_div_coeff * DIV + \
                   -args.gen_shell_coeff * SHELL

            z_g = torch.autograd.grad(gain, z_new)[0]
            z_new = (z_new + 1 * z_g).detach()


        if z_new_max is None:
            z_new_max = z_new.clone()
        else:
            z_new_max = torch.cat([z_new_max, z_new.clone()])

    z_new_max.require_grad = False

    if np.isnan(z_new_max.to('cpu').numpy()).any():
        mir_worked = 0
        mem_x = prev_gen.generate(args.batch_size*args.n_mem).detach()
    else:
        mem_x = prev_gen.decode(z_new_max).detach()
        mir_worked = 1

    return mem_x, mir_worked

def retrieve_hybrid(args, model, gen, prev_model, prev_gen, input_x, input_y, buffer, task):
    """ finds buffer samples whose latent space representation inteferes the most with the current batch """

    # first thing to do is actually to perform a gradient step on the current batch
    new_model = deepcopy(model)
    opt = torch.optim.SGD(new_model.parameters(), lr=0.1)

    opt.zero_grad()
    F.cross_entropy(new_model(input_x), input_y).backward()
    opt.step()

    # list of points to be returned
    hid_new_max = None

    assert args.max_loss_budget == 1, 'remove this if required. sanity check for now'

    for budget in range(args.max_loss_budget):

        with torch.no_grad():
            # encode the input using the previous model
            hid  = prev_gen.encode(input_x)
            hid_new = hid

        for step in range(args.max_loss_grad_steps):
            hid_new.requires_grad = True

            # decode into real image in order to classify
            x_dec = prev_gen.decode(hid_new)

            # get pseudo label from previous model
            y_old = prev_model(x_dec)

            # classify with the current network
            y_new = new_model(x_dec)

            # 1) KL cost
            KL = dist_kl(y_new, y_old)

            # 2) entropy term
            entropy = entropy_fn(y_old)
            if args.both_entropy: entropy += entropy_fn(y_new)

            # 3) diversity
            EUC = 0
            for found_h_i in range(budget if args.euc_coef > 0 else 0):
                EUC += F.mse_loss(hid_new, hid_new_max[found_h_i * hid_new.size(0):(found_h_i + 1) * hid_new.size(0)])

            gain = args.kl_coef * KL + args.ent_coef * entropy.sum() + args.euc_coef * EUC
            hid_g = torch.autograd.grad(gain, hid_new)[0]
             # TODO(10 is arbitrary)
            hid_new = (hid_new + 10 * hid_g).detach()

        if hid_new_max is None:
            hid_new_max = hid_new.detach()
        else:
            hid_new_max = torch.cat([hid_new_max, hid_new.detach()])

    h_int = hid_new_max

    with torch.no_grad():
        # 1 --> subsample the buffer
        # indices = torch.randperm(buffer.bx.size(0))# [:5 * args.buffer_batch_size]
        # bx, by, bt = buffer.bx[indices], buffer.by[indices], buffer.bt[indices]
        # TODO( 5 is arbitrary )
        bx, by, bt = buffer.sample(5 * args.buffer_batch_size , exclude_task = task)

        def get_dist(tensor_a, tensor_b):
            # assumes both are 2D tensors
            a_exp = tensor_a.unsqueeze(0).expand(tensor_b.size(0), -1, -1)
            b_exp = tensor_b.unsqueeze(1).expand(-1, tensor_a.size(0), -1)
            return ((a_exp - b_exp) ** 2).sum(dim=-1)

        dist = get_dist(bx, h_int)

        # let's do it w.r.t to the buffer instead
        # val, _ = dist.min(dim=0)

        # find the values with the smallest difference
        # _, argmin = val.topk(args.buffer_batch_size, largest=False, sorted=False)

        #k = args.buffer_batch_size // input_x.size(0)
        #_, argmin = dist.topk(k, largest=False, sorted=False)
        #argmin = argmin.flatten()

        _, argmin = dist.min(dim=1)
        out_x, out_y, out_t = bx[argmin], by[argmin], bt[argmin]

        n_repeats = args.buffer_batch_size // out_x.size(0)

        assert n_repeats > 0

        out_x = out_x.unsqueeze(0).expand(n_repeats, -1, -1).reshape(n_repeats * out_x.size(0), out_x.size(1))
        out_y = out_y.unsqueeze(0).expand(n_repeats, -1).reshape(n_repeats * out_y.size(0))
        out_t = out_t.unsqueeze(0).expand(n_repeats, -1).reshape(n_repeats * out_t.size(0))

        return out_x, out_y, out_t

def retrieve_replay_update(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    """ finds buffer samples with maxium interference """

    '''
    ER - MIR and regular ER
    '''


    updated_inds = None

    hid = model.return_hidden(input_x)

    logits = model.linear(hid)
    if args.multiple_heads:
        logits = logits.masked_fill(loader.dataset.mask == 0, -1e9)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    if args.method == 'mir_replay':
        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_task=task, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            logits_track_pre = model(bx)
            buffer_hid = model_temp.return_hidden(bx)
            logits_track_post = model_temp.linear(buffer_hid)

            if args.multiple_heads:
                mask = torch.zeros_like(logits_track_post)
                mask.scatter_(1, loader.dataset.task_ids[bt], 1)
                assert mask.nelement() // mask.sum() == args.n_tasks
                logits_track_post = logits_track_post.masked_fill(mask == 0, -1e9)
                logits_track_pre = logits_track_pre.masked_fill(mask == 0, -1e9)

            pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
            post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
            scores = post_loss - pre_loss
            EN_logits = entropy_fn(logits_track_pre)
            if args.compare_to_old_logits:
                old_loss = F.cross_entropy(buffer.logits[subsample], by,reduction="none")

                updated_mask = pre_loss < old_loss
                updated_inds = updated_mask.data.nonzero().squeeze(1)
                scores = post_loss - torch.min(pre_loss, old_loss)

            all_logits = scores
            big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]

            idx = subsample[big_ind]

        mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]
    else:
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_task=task)

    logits_buffer = model(mem_x)
    if args.multiple_heads:
        mask = torch.zeros_like(logits_buffer)
        mask.scatter_(1, loader.dataset.task_ids[b_task_ids], 1)
        assert mask.nelement() // mask.sum() == args.n_tasks
        logits_buffer = logits_buffer.masked_fill(mask == 0, -1e9)
    F.cross_entropy(logits_buffer, mem_y).backward()

    if updated_inds is not None:
        buffer.logits[subsample[updated_inds]] = deepcopy(logits_track_pre[updated_inds])
    opt.step()
    return model

'''OLD'''
def max_z_for_cls(args, virtual_cls, prev_cls, prev_gen, z_mu, z_var, z_t, gradient_steps=10, budget=10):

    z_new_max = None

    for i in range(budget):

        with torch.no_grad():

            if args.mir_init_prior:
                z_new = prev_gen.prior.sample((z_mu.shape[0],)).to(args.device)
            else:
                z_new = prev_gen.reparameterize(z_mu, z_var)

        for j in range(gradient_steps):

            z_new.requires_grad = True

            x_new = prev_gen.decode(z_new)
            y_pre = prev_cls(x_new)
            y_virtual = virtual_cls(x_new)

            # maximise the interference:
            XENT = 0
            if args.cls_xent_coeff>0.:
                XENT = cross_entropy(y_virtual, y_pre)

            # the predictions from the two models should be confident
            ENT = 0
            if args.cls_ent_coeff>0.:
                ENT = cross_entropy(y_pre, y_pre)
            #TODO(should we do the args.curr_entropy thing?)

            # the new found samples samples should be differnt from each others
            DIV = 0
            if args.cls_div_coeff>0.:
                for found_z_i in range(i):
                    DIV += F.mse_loss(
                        z_new,
                        z_new_max[found_z_i * z_new.size(0):found_z_i * z_new.size(0) + z_new.size(0)]
                        ) / (i)

            # (NEW) stay on gaussian shell loss:
            SHELL = 0
            if args.cls_shell_coeff>0.:
                SHELL = mse(torch.norm(z_new, 2, dim=1),
                        torch.ones_like(torch.norm(z_new, 2, dim=1))*np.sqrt(args.z_size))

            gain = args.cls_xent_coeff * XENT + \
                   -args.cls_ent_coeff * ENT + \
                   args.cls_div_coeff * DIV + \
                   -args.cls_shell_coeff * SHELL

            z_g = torch.autograd.grad(gain, z_new)[0]
            z_new = (z_new + 1 * z_g).detach()

        if z_new_max is None:
            z_new_max = z_new.clone()
        else:
            z_new_max = torch.cat([z_new_max, z_new.clone()])

    z_new_max.require_grad = False

    return z_new_max

def max_z_for_gen(args, virtual_gen, prev_gen, prev_cls, z_mu, z_var, z_t, gradient_steps=10, budget=10):
    """
    retrieve most interfered samples for the vae branch
    :param new_net:
    :param mu:
    :param logvar:
    :param z_t:
    :param gradient_steps:
    :param budget:
    :return:
    """
    z_new_max = None
    for i in range(budget):

        with torch.no_grad():

            if args.mir_init_prior:
                z_new = prev_gen.prior.sample((z_mu.shape[0],)).to(args.device)
            else:
                z_new = prev_gen.reparameterize(z_mu, z_var)

        for j in range(gradient_steps):
            z_new.requires_grad = True

            x_new = prev_gen.decode(z_new)


            prev_x_mean, prev_z_mu, prev_z_var, prev_ldj, prev_z0, prev_zk = \
                    prev_gen(x_new)
            _, prev_rec, prev_kl, _ = calculate_loss(prev_x_mean, x_new, prev_z_mu, \
                    prev_z_var, prev_z0, prev_zk, prev_ldj, args, beta=1)

            virtual_x_mean, virtual_z_mu, virtual_z_var, virtual_ldj, virtual_z0, virtual_zk = \
                    virtual_gen(x_new)
            _, virtual_rec, virtual_kl, _ = calculate_loss(virtual_x_mean, x_new, virtual_z_mu, \
                    virtual_z_var, virtual_z0, virtual_zk, virtual_ldj, args, beta=1)

            #TODO(warning, KL can explode)


            # maximise the interference
            KL = 0
            if args.gen_kl_coeff>0.:
                KL = virtual_kl - prev_kl

            REC = 0
            if args.gen_rec_coeff>0.:
                REC = virtual_rec - prev_rec

            # the predictions from the two models should be confident
            ENT = 0
            if args.gen_ent_coeff>0.:
                y_pre = prev_cls(x_new)
                ENT = cross_entropy(y_pre, y_pre)
            #TODO(should we do the args.curr_entropy thing?)

            DIV = 0
            # the new found samples samples should be differnt from each others
            if args.gen_div_coeff>0.:
                for found_z_i in range(i):
                    DIV += F.mse_loss(
                        z_new,
                        z_new_max[found_z_i * z_new.size(0):found_z_i * z_new.size(0) + z_new.size(0)]
                        ) / (i)

            # (NEW) stay on gaussian shell loss:
            SHELL = 0
            if args.gen_shell_coeff>0.:
                SHELL = mse(torch.norm(z_new, 2, dim=1),
                        torch.ones_like(torch.norm(z_new, 2, dim=1))*np.sqrt(args.z_size))

            gain = args.gen_kl_coeff * KL + \
                   args.gen_rec_coeff * REC + \
                   -args.gen_ent_coeff * ENT + \
                   args.gen_div_coeff * DIV + \
                   -args.gen_shell_coeff * SHELL

            z_g = torch.autograd.grad(gain, z_new)[0]
            z_new = (z_new + 1 * z_g).detach()


        if z_new_max is None:
            z_new_max = z_new.clone()
        else:
            z_new_max = torch.cat([z_new_max, z_new.clone()])

    z_new_max.require_grad = False

    return z_new_max
