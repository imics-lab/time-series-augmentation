#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""functions.py

helper functions used for GAN model training

Author: Xiaomin Li, Texas State University
Date: 10/20/2022


TODOS:
* put loss functions in seperate functions
"""

import os
import logging
import operator


import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

from torch.optim.optimizer import Optimizer


logger = logging.getLogger(__name__)


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

# weight init 
# check if we need weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if args.init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif args.init_type == 'orth':
            nn.init.orthogonal_(m.weight.data)
        elif args.init_type == 'xavier_uniform':
            nn.init.xavier_uniform(m.weight.data, 1.)
        else:
            raise NotImplementedError('{} unknown inital type'.format(args.init_type))
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# check zero grad 
def train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None):
    if args.GAN_type == 'TTS_TransferGAN':
        train_ttsTransferGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None)
    if args.GAN_type == 'TTS_GAN':
        train_ttsGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None)
    if args.GAN_type == 'TTS_CGAN':
        train_ttsCGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None)
    
def train_ttsTransferGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    
    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    
    # train_loader has two set of data, orginal signals and target signals
    for iter_idx, (org_sigs, org_labels, tag_sigs, tag_labels) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # original signals as generator input
        org_sigs = org_sigs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)

        # target signals as ground truths
        tag_sigs = tag_sigs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_validity = dis_net(tag_sigs)
        fake_sigs = gen_net(org_sigs).detach()
        
        assert fake_sigs.size() == tag_sigs.size(), f"fake_sigs.size(): {fake_sigs.size()} tag_sigs.size(): {tag_sigs.size()}"

        fake_validity = dis_net(fake_sigs)

        # cal loss
        if args.loss == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
            #soft label
            real_label = torch.full((tag_sigs.shape[0],), 0.9, dtype=torch.float, device=tag_sigs.get_device())
            fake_label = torch.full((tag_sigs.shape[0],), 0.1, dtype=torch.float, device=tag_sigs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=tag_sigs.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=tag_sigs.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=tag_sigs.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=tag_sigs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, tag_sigs, fake_sigs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, tag_sigs, fake_sigs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, tag_sigs, fake_sigs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss = d_loss/float(args.accumulated_times)  #accumulated_times = 1 default
        d_loss.backward()
        
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()

            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % (args.n_critic * args.accumulated_times) == 0:
            
            for accumulated_idx in range(args.g_accumulated_times):
                gen_sigs = gen_net(org_sigs)
                fake_validity = dis_net(gen_sigs)

                # cal loss
                loss_lz = torch.tensor(0)
                if args.loss == "standard":
                    real_label = torch.full((fake_validity.shape[0],), 1., dtype=torch.float, device=tag_sigs.get_device())
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
                elif args.loss == "lsgan":
                    real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=tag_sigs.get_device())
                    g_loss = nn.MSELoss()(fake_validity, real_label)
                elif args.loss == 'wgangp-mode':
                    fake_image1, fake_image2 = gen_sigs[:args.gen_batch_size//2], gen_sigs[args.gen_batch_size//2:]
                    z_random1, z_random2 = gen_z[:args.gen_batch_size//2], gen_z[args.gen_batch_size//2:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                    torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)

                    g_loss = -torch.mean(fake_validity) + loss_lz
                else:
                    g_loss = -torch.mean(fake_validity)
                    
                g_loss = g_loss/float(args.g_accumulated_times)
                g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # adjust learning rate
            # where does it change the learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
            gen_step += 1

        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1 

        
        
def train_ttsGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        
        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_validity = dis_net(fake_imgs)

        # cal loss
        if args.loss == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
            #soft label
            real_label = torch.full((imgs.shape[0],), 0.9, dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0.1, dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss = d_loss/float(args.accumulated_times)
        d_loss.backward()
        
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()

            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % (args.n_critic * args.accumulated_times) == 0:
            
            for accumulated_idx in range(args.g_accumulated_times):
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
                gen_imgs = gen_net(gen_z)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                loss_lz = torch.tensor(0)
                if args.loss == "standard":
                    real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
                if args.loss == "lsgan":
                    if isinstance(fake_validity, list):
                        g_loss = 0
                        for fake_validity_item in fake_validity:
                            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                            g_loss += nn.MSELoss()(fake_validity_item, real_label)
                    else:
                        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                        # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                        g_loss = nn.MSELoss()(fake_validity, real_label)
                elif args.loss == 'wgangp-mode':
                    fake_image1, fake_image2 = gen_imgs[:args.gen_batch_size//2], gen_imgs[args.gen_batch_size//2:]
                    z_random1, z_random2 = gen_z[:args.gen_batch_size//2], gen_z[args.gen_batch_size//2:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                    torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)

                    g_loss = -torch.mean(fake_validity) + loss_lz
                else:
                    g_loss = -torch.mean(fake_validity)
                g_loss = g_loss/float(args.g_accumulated_times)
                g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # adjust learning rate
            # where does it change the learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
            gen_step += 1

        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1 



def train_ttsCGAN(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    cls_criterion = nn.CrossEntropyLoss()
    lambda_cls = 1
    lambda_gp = 10
    
    # train mode
    gen_net.train()
    dis_net.train()
    
    for iter_idx, (real_imgs, real_img_labels) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        
        # Adversarial ground truths
        real_imgs = real_imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
#         real_img_labels = real_img_labels.type(torch.IntTensor)
        real_img_labels = real_img_labels.type(torch.LongTensor)
        real_img_labels = real_img_labels.cuda(args.gpu, non_blocking=True)

        # Sample noise as generator input
        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).cuda(args.gpu, non_blocking=True)
        fake_img_labels = torch.randint(0, 5, (real_imgs.shape[0],)).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        dis_net.zero_grad()
        r_out_adv, r_out_cls = dis_net(real_imgs)
        fake_imgs = gen_net(noise, fake_img_labels)
        
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        f_out_adv, f_out_cls = dis_net(fake_imgs)

        # Compute loss for gradient penalty.
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).cuda(args.gpu, non_blocking=True)  # bh, C, H, W
        x_hat = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
        out_src, _ = dis_net(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, args)
        
        d_real_loss = -torch.mean(r_out_adv)
        d_fake_loss = torch.mean(f_out_adv)
        d_adv_loss = d_real_loss + d_fake_loss 
        
        d_cls_loss = cls_criterion(r_out_cls, real_img_labels)
        
        d_loss = d_adv_loss + lambda_cls * d_cls_loss + lambda_gp * d_loss_gp
        d_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        
        gen_net.zero_grad()

        gen_imgs = gen_net(noise, fake_img_labels)
        g_out_adv, g_out_cls = dis_net(gen_imgs)

        g_adv_loss = -torch.mean(g_out_adv)
        g_cls_loss = cls_criterion(g_out_cls, fake_img_labels)    
        g_loss = g_adv_loss + lambda_cls * g_cls_loss
        g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        gen_optimizer.step()

        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            writer.add_scalar('LR/g_lr', g_lr, global_steps)
            writer.add_scalar('LR/d_lr', d_lr, global_steps)

        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.dis_batch_size * args.world_size * global_steps
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
            del cpu_p

        writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
        gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), ema_beta))

        del gen_imgs
        del real_imgs
        del fake_imgs
        del f_out_adv
        del r_out_adv
        del r_out_cls
        del g_out_cls
        del g_cls_loss
        del g_adv_loss
        del g_loss
        del d_cls_loss
        del d_adv_loss
        del d_loss

        writer_dict['train_global_steps'] = global_steps + 1 


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr



class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss