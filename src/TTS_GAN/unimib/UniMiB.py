#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UniMiB.py

PyTorch dataloaders for UniMiB dataset

Author: Bikram De, Texas State University
Date: 10/19/2022


TODOS:
* 
"""


#necessory import libraries

import os 
import sys 
import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#class names and corresponding labels of the UniMiB dataset


class unimib_oneClass(Dataset):
    def __init__(self, 
        verbose = False,
        incl_xyz_accel = True, #include component accel_x/y/z in ____X data
        incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                               #False => combine test & validation groups
        is_normalize = False,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        one_hot_encode = True, data_mode = 'Train', single_class = True, class_name= 'Running', augment_times = None, class_id = 0):
        
        self.verbose = verbose
        self.incl_xyz_accel = incl_xyz_accel
        self.incl_rms_accel = incl_rms_accel
        self.incl_val_group = incl_val_group
        self.split_subj = split_subj
        self.one_hot_encode = one_hot_encode
        self.data_mode = data_mode
        self.class_name = class_name
        self.class_id = class_id
        self.single_class = single_class
        self.is_normalize = is_normalize
    

        self.x_train, self.y_train, self.x_validate, self.y_validate, \
        self.x_test, self.y_test = self.unimib_load_dataset(incl_val_group = True)
        #print(self.x_train.shape, self.y_train.shape)
        if self.single_class:
            self.one_class_train_data, self.one_class_train_labels = \
            self.extract_one_class(self.class_id, self.x_train, self.y_train)
        # if reshape:
        print(self.one_class_train_data.shape)
        self.one_class_train_data = self.one_class_train_data.reshape\
        (self.one_class_train_data.shape[0], self.one_class_train_data.shape[2], 1, self.one_class_train_data.shape[1])

    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def get_shapes(np_arr_list):
        """Returns text, each line is shape and dtype for numpy array in list
           example: print(get_shapes([X_train, X_test, y_train, y_test]))"""
        shapes = ""
        for i in np_arr_list:
            my_name = namestr(i,globals())
            shapes += (my_name[0] + " shape is " + str(i.shape) \
                + " data type is " + str(i.dtype) + "\n")
        return shapes
    
    def extract_one_class(self, class_id, x_train, y_train):
        # class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,\
        #               'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}
        one_class_train_data = []
        one_class_train_labels = []
        for i, label in enumerate(y_train):
            class_label = np.argmax(label)
            if class_label == class_id:
                one_class_train_data.append(x_train[i])
                one_class_train_labels.append(label)
        one_class_train_data = np.array(one_class_train_data)
        one_class_train_labels = np.array(one_class_train_labels)
        return one_class_train_data, one_class_train_labels

    def unimib_load_dataset(self,
        verbose = True,
        incl_xyz_accel = True, #include component accel_x/y/z in ____X data
        incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
        incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                               #False => combine test & validation groups
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        one_hot_encode = True):
        #Download and unzip original dataset
        if (not os.path.isfile('./UniMiB-SHAR.zip')):
            print("Downloading UniMiB-SHAR.zip file")
            #invoking the shell command fails when exported to .py file
            #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
        if (not os.path.isdir('./UniMiB-SHAR')):
            shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
        #Convert .mat files to numpy ndarrays
        path_in = 'UniMiB-SHAR/data/'
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']
    
        if(verbose):
            headers = ("Raw data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Reshape data and compute total (rms) acceleration
        num_samples = 151 
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        if (incl_rms_accel):
            rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
            adl_data = np.dstack((adl_data,rms_accel))
        #remove component accel if needed
        if (not incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
        if(verbose):
            headers = ("Reshaped data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html
    
    
        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels
    
        if (not incl_val_group):
            train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] + 
                                            split_subj['validation_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]
        else:
            train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
            x_train = adl_data[train_index]
            y_train = act_num[train_index]
    
            validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
            x_validation = adl_data[validation_index]
            y_validation = act_num[validation_index]
    
        test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]
    
        if (verbose):
            print("x/y_train shape ",x_train.shape,y_train.shape)
            if (incl_val_group):
                print("x/y_validation shape ",x_validation.shape,y_validation.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)
        #If selected one-hot encode y_* using keras to_categorical, reference:
        #https://keras.io/api/utils/python_utils/#to_categorical-function and
        #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
        if (one_hot_encode):
            y_train = to_categorical(y_train, num_classes=9)
            if (incl_val_group):
                y_validation = to_categorical(y_validation, num_classes=9)
            y_test = to_categorical(y_test, num_classes=9)
            if (verbose):
                print("After one-hot encoding")
                print("x/y_train shape ",x_train.shape,y_train.shape)
                if (incl_val_group):
                    print("x/y_validation shape ",x_validation.shape,y_validation.shape)
                print("x/y_test shape  ",x_test.shape,y_test.shape)
        if (incl_val_group):
            return x_train, y_train, x_validation, y_validation, x_test, y_test
        else:
            return x_train, y_train, x_test, y_test

    
    def __len__(self):
        
        if self.data_mode == 'Train':
            if self.single_class:
                return len(self.one_class_train_labels)
            else:
                return len(self.y_train)
        else:
            if self.single_class:
                return len(self.one_class_test_labels)
            else:
                return len(self.y_test)
        
    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            if self.single_class:
                return self.one_class_train_data[idx], self.one_class_train_labels[idx]
            else:
                return self.x_train[idx], self.y_train[idx]
        else:
            if self.single_class:
                return self.one_class_test_data[idx], self.one_class_test_labels[idx]
            else:
                return self.x_test[idx], self.y_test[idx]
# class mitbih_oneClass(Dataset):
#     """
#     A pytorch dataloader loads on class data from mithib_train dataset.
#     Example Usage:
#         class0 = mitbih_oneClass(class_id = 0)
#         class1 = mitbih_oneClass(class_id = 1)
#     """
#     def __init__(self, filename='./mitbih_train.csv', reshape = True, class_id = 0):
#         data_pd = pd.read_csv(filename, header=None)
#         data = data_pd[data_pd[187] == class_id]
    
#         self.data = data.iloc[:, :-1].values
#         self.labels = data[187].values
        
#         if reshape:
#             self.data = self.data.reshape(self.data.shape[0], 1, 1, self.data.shape[1])
        
# #         print(f'Data shape of {reverse_cls_dit[class_id]} instances = {self.data.shape}')
        
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]
        

        
def main():
    class_i = unimib_oneClass(class_id = class_id) #mitbih_oneClass(class_id = 0)
    
        
if __name__ == "__main__":
    main()