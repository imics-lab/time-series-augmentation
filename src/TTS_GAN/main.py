#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main.py

Style Transfor Transformer model training starts here

Author: Xiaomin Li, Texas State University
Date: 10/20/2022


TODOS:
* create training process
* Remove unnecessary statements

* make load from checkpoint work
"""

import cfg
from GANModels import Generator, Generator_z, Discriminator 
from getDataLoader import getDataLoader
from functions import train, LinearLrDecay, weights_init, AdamW
from utils import set_log_dir, save_checkpoint, create_logger, load_params, copy_params, cur_stages, gen_plot

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import PIL.Image
from copy import deepcopy
import random 

GAN_types = ["TTS_GAN", "TTS_CGAN", "TTS_TransferGAN"]

def main():
    args = cfg.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    #check if this statement is executed
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    assert args.GAN_type in GAN_types
    
    # import network
    # set parameters in parse arguments
    if args.GAN_type == 'TTS_TransferGAN':
        gen_net = Generator(seq_len=args.g_seq_len, channels=args.g_channel, data_embed_dim=args.g_data_embed_dim)
        print(gen_net)
        dis_net = Discriminator(in_channels=args.d_channel, patch_size=args.d_patch_size, emb_size=args.d_emb_size, seq_length = args.d_seq_length, n_classes=args.d_n_classes)
        print(dis_net)
    if args.GAN_type == 'TTS_GAN':
        gen_net = Generator_z(seq_len=args.g_seq_len, channels=args.g_channel, data_embed_dim=args.g_data_embed_dim)
        #gen_net = Generator_z()
        print(gen_net)
        dis_net = Discriminator(in_channels=args.d_channel, patch_size=args.d_patch_size, emb_size=args.d_emb_size, seq_length = args.d_seq_length, n_classes=args.d_n_classes)
        #dis_net = Discriminator()
        print(dis_net)
    if args.GAN_type == 'TTS_CGAN':
        pass
    
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpu)
        gen_net.cuda(args.gpu)
        dis_net.cuda(args.gpu)
        

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.g_lr, weight_decay=args.wd)
        
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)
    
    #get dataloader from different datasets
    train_loader = getDataLoader(args)
    
    
    
    if args.max_iter:
        max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader)) 
        

    # initial
    # does not need fixed_z 
    # why deepcopy model params?
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    # check the proper way to load checkpoints
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        # check loaded paramters
        best_fid = checkpoint['best_fid']
        
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
        # why do this? 
#         avg_gen_net = deepcopy(gen_net)
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        del avg_gen_net
        gen_avg_param = list(p.cuda().to(f"cuda:{args.gpu}") for p in gen_avg_param)
        
        
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
    # create new log dir 
        assert args.exp_name # set logger name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])
    
    if args.rank == 0:
        logger.info(args)
        
    #figure out how to use writer_dict
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(max_epoch)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        # check what does it print, is it useful?
#         print("cur_stage " + str(cur_stage)) if args.rank==0 else 0  # print the same thing
        print(f"path: {args.path_helper['prefix']}") if args.rank==0 else 0

        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        
#         if (epoch+1) % args.val_freq == 0:
        # plot synthetic data in tensorboard
        gen_net.eval()
        with torch.no_grad():
            plot_buf = gen_plot(gen_net, epoch, args)
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image).unsqueeze(0)
            writer.add_image('Image', image[0], epoch)
        
        # why deepcopy net params? 
        is_best = False
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
#             'gen_state_dict': gen_net.module.state_dict(),
#             'dis_state_dict': dis_net.module.state_dict(),
#             'avg_gen_state_dict': avg_gen_net.module.state_dict(),
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
#             'best_fid': best_fid,
            'path_helper': args.path_helper,
#             'fixed_z': fixed_z
        }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net


if __name__ == '__main__':
    main()
