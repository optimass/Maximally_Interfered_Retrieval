import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args
        self.k    = 0.03

        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        # TODO(change this:)
        if args.gen:
            if 'mnist' in args.dataset:
                img_size = 784
                economy = img_size // input_size[0]
            elif 'cifar' in args.dataset:
                img_size = 32 * 32 * 3
                economy = img_size // (input_size[0] ** 2)
            elif 'imagenet' in args.dataset:
                img_size = 84 * 84 * 3
                economy = img_size // (input_size[0] ** 2)
        else:
            economy = 1

        buffer_size = economy  * args.mem_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)
        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)

        if args.cuda:
            bx = bx.to(args.device)
            by = by.to(args.device)
            bt = bt.to(args.device)
            logits = logits.to(args.device)

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)

        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        #Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)


            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.by.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.bt.size(0), pdb.set_trace()

        assert idx_new_data.max() < x.size(0), pdb.set_trace()
        assert idx_new_data.max() < y.size(0), pdb.set_trace()

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]


    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno    = one_hot_y.sum(dim=0)
            per_class_acc     = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc    = per_class_acc
            self.valid_deno   = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]


    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.br = self.bt[:remove_after_this_idx]

    def sample(self, amt, exclude_task = None, ret_ind = False):
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bt = self.bx[valid_indices], self.by[valid_indices], self.bt[valid_indices]
        else:
            bx, by, bt = self.bx[:self.current_index], self.by[:self.current_index], self.bt[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, bt, torch.from_numpy(np.arange(bx.size(0)))
            else:
                return bx, by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

            if self.args.cuda:
                indices = indices.to(self.args.device)

            if ret_ind:
                return bx[indices], by[indices], bt[indices], indices
            else:
                return bx[indices], by[indices], bt[indices]

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]


def get_cifar_buffer(args, hH=8, gen=None):
    args.input_size = (hH, hH)
    args.gen = True

    return Buffer(args, gen=gen)
