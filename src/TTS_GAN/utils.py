#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""utils.py

helper functions used for GAN model training

Author: Xiaomin Li, Texas State University
Date: 10/20/2022


TODOS:
* Remove unnecessary code
"""

import logging
import os
import io
import time
import torch
import dateutil.tz
import numpy as np
from datetime import datetime
from copy import deepcopy
#from MITBIH import mitbih_oneClass, mitbih_twoClass

import matplotlib.pyplot as plt
from torch.utils import data

    
def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))
        

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx
    
def gen_plot(gen_net, epoch, args):
    if args.GAN_type == 'TTS_TransferGAN':
        buf = gen_plot_ttsTransferGAN(gen_net, epoch, args)
    if args.GAN_type == 'TTS_GAN':
        buf = gen_plot_ttsGAN(gen_net, epoch, args)
    if args.GAN_type == 'TTS_CGAN':
        buf = gen_plot_ttsCGAN(gen_net, epoch, args)
    
    return buf
        
def gen_plot_ttsTransferGAN(gen_net, epoch, args):    
    """Create a pyplot plot and save to buffer."""
    
    synthetic_data = [] 
    
    org_signal = mitbih_oneClass(class_id = 0)
    org_data_loader = data.DataLoader(org_signal, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    for i, (sigs, labels) in enumerate(org_data_loader):
        sigs = sigs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        if i < 10:
            fake_sigs = gen_net(sigs)
            fake_sigs = fake_sigs.to('cpu').detach().numpy()
            synthetic_data.append(fake_sigs)
        else:
            break

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic data at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


def gen_plot_ttsGAN(gen_net, epoch, args):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 

    for i in range(10):
        fake_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 100))).cuda(args.gpu, non_blocking=True)
        fake_sigs = gen_net(fake_noise)
        fake_sigs = fake_sigs.to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic data at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

def gen_plot_ttsCGAN(gen_net, epoch, args):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 
    synthetic_labels = []
    
    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
        fake_label = torch.randint(0, 5, (1,))
        fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()
        
        synthetic_data.append(fake_sigs)
        synthetic_labels.append(fake_label)

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic data at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
            axs[i, j].title.set_text(synthetic_labels[i*5+j])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


