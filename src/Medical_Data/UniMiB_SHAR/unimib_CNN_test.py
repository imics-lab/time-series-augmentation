# -*- coding: utf-8 -*-
"""UniMiB_SHAR_ADL_load_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U1EY6cZsOFERD3Df1HRqjuTq5bDUGH03

#UniMiB_SHAR_ADL_load_dataset.ipynb. 
Loads the A-9 (ADL) portion of the UniMiB dataset from the Internet repository and converts the data into numpy arrays while adhering to the general format of the [Keras MNIST load_data function](https://keras.io/api/datasets/mnist/#load_data-function).

Arguments: tbd
Returns: Tuple of Numpy arrays:   
(x_train, y_train),(x_validation, y_validation)\[optional\],(x_test, y_test) 

* x_train\/validation\/test: containing float64 with shapes (num_samples, 151, {3,4,1})
* y_train\/validation\/test: containing int8 with shapes (num_samples 0-9)

The train/test split is by subject

Example usage:  
x_train, y_train, x_test, y_test = unimib_load_dataset()

Additional References  
 If you use the dataset and/or code, please cite this paper (downloadable from [here](http://www.mdpi.com/2076-3417/7/10/1101/html))

Developed and tested using colab.research.google.com  
To save as .py version use File > Download .py

Author:  Lee B. Hinkle, IMICS Lab, Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.


TODOs:
* Fix document strings
* Assign names to activities instead of numbers
"""
from sklearn.utils import shuffle

import tensorflow as tf
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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



#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unimib_load_dataset(
    verbose = True,
    incl_xyz_accel = True, #include component accel_x/y/z in ____X data
    incl_rms_accel = True, #add rms value (total accel) of accel_x/y/z in ____X data
    incl_val_group = True, #True => returns x/y_test, x/y_validation, x/y_train
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
    path_in = './UniMiB-SHAR/data'
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

def train_model(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
    #dls.dataset
    model = build_ts_model(InceptionTimePlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(20, lr_max=1e-2)
    return learn

def run_model(x_test, y_test, learn):
    probas, target, preds = learn.get_X_preds(x_test, y_test)
    preds_labels = np.argmax(preds, axis=1)
    target_labels = np.argmax(target, axis=-1) # undo one-hot encoding
    return preds_labels, target_labels

def jitter(x, sigma):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
def scaling(x, sigma):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])    

def magnitude_warp(x, sigma, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret    

def time_warp(x, sigma, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_warp(x, window_ratio, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def window_slice(x, reduce_ratio):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def testing_values(x_train_total, y_train_total, augment_values_list, augment_func):
    acc_list_mean = []
    for augment_value in augment_values_list:
        acc_list_temp = []
        for shuffle_id in range(10):
            x_train_new, x_validate, y_train_new, y_validate = train_test_split(\
                x_train_total, y_train_total, test_size = 0.25, shuffle = True)
            if (augment_func == 'Jitter'):
                x_train_augment_initial = jitter(x_train_new, augment_value)
            elif (augment_func == 'Scale'):
                x_train_augment_initial = scaling(x_train_new, augment_value)
            elif (augment_func == 'MagWarp'):
                x_train_augment_initial = magnitude_warp(x_train_new, augment_value, 4)
            elif (augment_func == 'TimeWarp'):
                x_train_augment_initial = time_warp(x_train_new, augment_value, 4)
            elif (augment_func == 'WinWarp'):
                x_train_augment_initial = window_warp(x_train_new, augment_value, 4)
            elif (augment_func == 'WinSlice'):
                x_train_augment_initial = window_slice(x_train_new, augment_value)
            
            x_train_augment = np.concatenate((x_train_new, x_train_augment_initial), axis = 0)
            y_train_augment = np.concatenate((y_train_new, y_train_new), axis = 0)
            x_train_shuffled, y_train_shuffled = shuffle(x_train_augment, y_train_augment)
            learn = train_model(x_train_shuffled, y_train_shuffled, x_validate, y_validate)
            preds_labels, target_labels = run_model(x_test, y_test, learn)
            acc_list_temp.append(accuracy_score(target_labels, preds_labels))
        acc_list_mean.append(np.mean(acc_list_temp))

    optimal_augment_value = augment_values_list[np.argmax(acc_list_mean)]
    
    return optimal_augment_value

if __name__ == "__main__":
    print("Downloading and processing UniMiB SHAR dataset, ADL Portion")
    x_train, y_train, x_validation, y_validation, x_test, y_test = unimib_load_dataset()
    print("\nUniMiB SHAR returned arrays:")
    print("x_train shape ",x_train.shape," y_train shape ", y_train.shape)
    print("x_valid shape ",x_validation.shape, "y_validation shape ", y_validation.shape)
    print("x_test shape  ",x_test.shape," y_test shape  ",y_test.shape)

    x_train_total = np.concatenate((x_train, x_validation), axis = 0)
    y_train_total = np.concatenate((y_train, y_validation), axis = 0)
    print(x_train_total.shape, y_train_total.shape)
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    print('JITTERING')
    jitter_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 0.8, 1]
    jitter_opt = testing_values(x_train_total, y_train_total, jitter_values, 'Jitter')
    print(jitter_opt)
    
    print('SCALING')
    scale_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    scale_opt = testing_values(x_train_total, y_train_total, scale_values, 'Scale')
    print(scale_opt)

    print('MAGNITUDE WARPING')
    mag_warp_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    mag_warp_opt = testing_values(x_train_total, y_train_total, mag_warp_values, 'MagWarp')
    print(mag_warp_opt)

    print('TIME WARPING')
    time_warp_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    time_warp_opt = testing_values(x_train_total, y_train_total, time_warp_values, 'TimeWarp')
    print(time_warp_opt)
    
    print('WINDOW WARPING')
    window_warp_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]
    window_warp_opt = testing_values(x_train_total, y_train_total, window_warp_values, 'WinWarp')
    print(window_warp_opt)
    
    print('WINDOW SLICING')
    window_slice_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    window_slice_opt = testing_values(x_train_total, y_train_total, window_slice_values, 'WinSlice')
    print(window_slice_opt)
    
    print('InceptionTime')
    final_values_dict = {'Jitter': jitter_opt,\
                'Scale': scale_opt,\
                'Magnitude Warp': mag_warp_opt,\
                'Time Warp': time_warp_opt,\
                'Window Warp': window_warp_opt,\
                'Window Slicing': window_slice_opt}

    print(final_values_dict)
    

    
    