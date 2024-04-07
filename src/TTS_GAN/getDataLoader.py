#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""getDataLoader.py

get pytorch dataloader by input argument dataset name

Author: Xiaomin Li, Texas State University
Date: 11/3/2022


TODOS:
* add more datasets
"""


# from MITBIH import mitbih_oneClass, mitbih_twoClass
# from UniMiB import unimib_oneClass
#from leotta import leotta_oneClass
#from psg import psg_oneClass
#from mitbih import mitbih_oneClass
from eeg import eeg_oneClass
from torch.utils import data
import sys

datasets = ['MITBIH', 'UNIMIB', 'LEOTTA', 'PSG', 'EEG']

def getDataLoader(args):
    
    assert args.dataset in datasets
    class_name = args.class_name
    
    if args.GAN_type == 'TTS_GAN':
        #oneClassdata = unimib_oneClass(class_id = args.class_id)
        #oneClassdata = leotta_oneClass(class_id = args.class_id)
        #oneClassdata = psg_oneClass(class_id = args.class_id)
        #oneClassdata = mitbih_oneClass(class_id = args.class_id)
        oneClassdata = eeg_oneClass(class_id = args.class_id)
        train_loader = data.DataLoader(oneClassdata, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # if args.GAN_type == 'TTS_TransferGAN':
    #     twoClassdata = mitbih_twoClass(class_id1 = 0, class_id2 = 1) #data_1, labels_1, data_2, labels_2
    #     train_loader = data.DataLoader(twoClassdata, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # if args.GAN_type == 'TTS_GAN':
    #     oneClassdata = mitbih_oneClass(class_id = 0)
    #     train_loader = data.DataLoader(oneClassdata, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # if args.GAN_type == 'TTS_CGAN':
    #     pass

    
    return train_loader
        
    
    