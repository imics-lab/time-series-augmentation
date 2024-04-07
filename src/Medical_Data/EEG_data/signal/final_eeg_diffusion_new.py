# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import sys
# sys.path.insert(0, '../../src/signal/')
# sys.path.insert(0, '../../src/signal/modules/')
from modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free

from sklearn.utils import shuffle
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
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

def train_model_lstm(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[2048, 512])
    #dls.dataset
    model = build_ts_model(LSTM_FCNPlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(10, lr_max=1e-2)
    return learn

def train_model_cnn(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
    #dls.dataset
    model = build_ts_model(InceptionTimePlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(10, lr_max=1e-2)
    return learn

def train_model_tst(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
    #dls.dataset
    model = build_ts_model(TSTPlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(10, lr_max=1e-2)
    return learn

def run_model(x_test, y_test, learn):
    probas, target, preds = learn.get_X_preds(x_test, y_test)
    preds_labels = np.argmax(preds, axis=1)
    target_labels = np.argmax(target, axis=-1) # undo one-hot encoding
    return preds_labels, target_labels


def extract_one_class(class_id):
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


if __name__ == "__main__":
    input_dir = '../eeg_dataset_1/'
    x_train = np.load(input_dir +'x_train.npy')
    x_validation = np.load(input_dir +'x_valid.npy')
    x_test = np.load(input_dir +'x_test.npy')
    y_train = np.load(input_dir +'y_train.npy')
    y_validation = np.load(input_dir +'y_valid.npy')
    y_test = np.load(input_dir +'y_test.npy')
    
    x_train = x_train.astype(float)
    x_validation = x_validation.astype(float)
    x_test = x_test.astype(float)

    x_train = x_train[:, :4096,:]
    x_validation = x_validation[:, :4096,:]
    x_test = x_test[:, :4096,:]
    
    # x_train = x_train.reshape\
    # (x_train.shape[0], x_train.shape[2], x_train.shape[1])

    # x_validation = x_validation.reshape\
    # (x_validation.shape[0], x_validation.shape[2], x_validation.shape[1])

    # x_test = x_test.reshape\
    # (x_test.shape[0], x_test.shape[2], x_test.shape[1])
    
    x_train_new = []
    y_train_new = []
    device = "cuda:3"
    model = Unet1D_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = y_train.shape[1],
        cond_drop_prob = 0.5,
        channels = 1)
    ckpt = torch.load("checkpoint/DDPM1D_cls_free_EEG/checkpoint.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    diffusion = GaussianDiffusion1D_cls_free(
            model,
            seq_length = 4096,
            timesteps = 1000).to(device)
    for class_id in range(y_train.shape[1]):
        x_train_id, y_train_id = extract_one_class(class_id)
        #x_train_id = x_train_id.reshape(x_train_id.shape[0], x_train_id.shape[2], series_length)
        
        sample_size = x_train_id.shape[0]

        y = torch.Tensor([class_id] * sample_size).long().to(device)
        x_train_id_new = diffusion.sample(classes = y, cond_scale = 3.).cpu().numpy()
        x_train_id_new = x_train_id_new.reshape(x_train_id_new.shape[0], x_train_id_new.shape[2], x_train_id_new.shape[1])
        y_train_id_new = [y_train_id[0]]*sample_size
        y_train_id_new = np.array(y_train_id_new)
        print(x_train_id_new.shape, y_train_id_new.shape)
        
        
        x_train_augment = np.concatenate([x_train_id_new, x_train_id], axis=0)
        y_train_augment = np.concatenate([y_train_id_new, y_train_id], axis=0)
        x_train_new.append(x_train_augment)
        y_train_new.append(y_train_augment)
        #print(x_train_new.shape, y_train_new.shape)

    x_train_new = np.concatenate(x_train_new, axis=0)
    y_train_new = np.concatenate(y_train_new, axis=0)
    #x_train_new = x_train_new.reshape(x_train_new.shape[0], series_length, x_train_new.shape[1])
    print(x_train_new.shape, y_train_new.shape)
    x_train_new, y_train_new = shuffle(x_train_new, y_train_new, random_state=42)

    np.save('x_train_diff', x_train_new)
    np.save('y_train_diff', y_train_new)
    np.save('x_valid', x_valid)
    np.save('y_valid', y_valid)
    np.save('x_test', x_test)
    np.save('y_test', y_test)

    learn_lstm = train_model_lstm(x_train_new, y_train_new, x_validation, y_validation)
    preds_labels_lstm, target_labels_lstm = run_model(x_test, y_test, learn_lstm)
    acc_lstm = accuracy_score(target_labels_lstm, preds_labels_lstm)

    learn_cnn = train_model_cnn(x_train_new, y_train_new, x_validation, y_validation)
    preds_labels_cnn, target_labels_cnn = run_model(x_test, y_test, learn_cnn)
    acc_cnn = accuracy_score(target_labels_cnn, preds_labels_cnn)

    learn_tst = train_model_tst(x_train_new, y_train_new, x_validation, y_validation)
    preds_labels_tst, target_labels_tst = run_model(x_test, y_test, learn_tst)
    acc_tst = accuracy_score(target_labels_tst, preds_labels_tst)

    print('LSTM_FCN: ', acc_lstm)
    print('InceptionTime: ', acc_cnn)
    print('TST: ', acc_tst)
