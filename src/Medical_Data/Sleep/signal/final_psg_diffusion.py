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

def load_psg_data():
    input_dir = '../PSG/'
    x_train = np.load(input_dir +'x_train.npy')
    x_validation = np.load(input_dir +'x_valid.npy')
    x_test = np.load(input_dir +'x_test.npy')
    y_train = np.load(input_dir +'y_train.npy')
    y_validation = np.load(input_dir +'y_valid.npy')
    y_test = np.load(input_dir +'y_test.npy')
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def train_model_lstm(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
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
    learn.fit_one_cycle(20, lr_max=1e-2)
    return learn

def train_model_tst(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
    #dls.dataset
    model = build_ts_model(TSTPlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(25, lr_max=1e-3)
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
    print("Get PSG-Audio (npy data)")
    x_train, y_train, x_valid, y_valid, x_test, y_test \
                             = load_psg_data()
    headers = ("Array","shape", "data type")
    mydata = [("x_train:", x_train.shape, x_train.dtype),
            ("y_train:", y_train.shape, y_train.dtype),
            ("x_valid:", x_valid.shape, x_valid.dtype),
            ("y_valid:", y_valid.shape, y_valid.dtype),
            ("x_test:", x_test.shape, x_test.dtype),
            ("y_test:", y_test.shape, y_test.dtype)]
    print("\n",tabulate(mydata, headers=headers))
    print ('\n','-'*72) # just a dashed line
    # x_train_total = np.concatenate((x_train, x_validation), axis = 0)
    # y_train_total = np.concatenate((y_train, y_validation), axis = 0)
    # print(x_train_total.shape, y_train_total.shape)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    pad_width = [(0, 0), (0, 512 - x_train.shape[1]), (0, 0)]
    # Pad the array with zeros
    x_train = np.pad(x_train, pad_width, mode='constant', constant_values=0)
    x_valid = np.pad(x_valid, pad_width, mode='constant', constant_values=0)
    x_test = np.pad(x_test, pad_width, mode='constant', constant_values=0)
    #sample_size = 1000
    x_train_new = []
    y_train_new = []
    device = "cuda:3"
    model = Unet1D_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = y_train.shape[1],
        cond_drop_prob = 0.5,
        channels = x_train.shape[2])
    ckpt = torch.load("checkpoint/DDPM1D_cls_free_PSG/checkpoint.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    diffusion = GaussianDiffusion1D_cls_free(
            model,
            seq_length = 512,
            timesteps = 1000).to(device)
    for class_id in range(y_train.shape[1]):
        x_train_id, y_train_id = extract_one_class(class_id)
        #x_train_id = x_train_id.reshape(x_train_id.shape[0], x_train_id.shape[2], series_length)
        
        sample_size = int(x_train_id.shape[0]/10)
        x_train_augment = None
        y_train_augment = None
        for i in range(10):

            y = torch.Tensor([class_id] * sample_size).long().to(device)
            x_train_id_new = diffusion.sample(classes = y, cond_scale = 3.).cpu().numpy()
            x_train_id_new = x_train_id_new.reshape(x_train_id_new.shape[0], x_train_id_new.shape[2], x_train_id_new.shape[1])
            y_train_id_new = [y_train_id[0]]*sample_size
            y_train_id_new = np.array(y_train_id_new)
            print(x_train_id_new.shape, y_train_id_new.shape)
            if x_train_augment is None:
                x_train_augment = x_train_id_new
            else:
                x_train_augment = np.concatenate([x_train_id_new, x_train_augment], axis=0)
        
            if y_train_augment is None:
                y_train_augment = y_train_id_new
            else:
                y_train_augment = np.concatenate([y_train_id_new, y_train_augment], axis=0)
            
        x_train_augment = np.concatenate([x_train_id, x_train_augment], axis=0)
        y_train_augment = np.concatenate([y_train_id, y_train_augment], axis=0)
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

    # learn_lstm = train_model_lstm(x_train_new, y_train_new, x_valid, y_valid)
    # preds_labels_lstm, target_labels_lstm = run_model(x_test, y_test, learn_lstm)
    # acc_lstm = accuracy_score(target_labels_lstm, preds_labels_lstm)

    # learn_cnn = train_model_cnn(x_train_new, y_train_new, x_valid, y_valid)
    # preds_labels_cnn, target_labels_cnn = run_model(x_test, y_test, learn_cnn)
    # acc_cnn = accuracy_score(target_labels_cnn, preds_labels_cnn)

    # learn_tst = train_model_tst(x_train_new, y_train_new, x_valid, y_valid)
    # preds_labels_tst, target_labels_tst = run_model(x_test, y_test, learn_tst)
    # acc_tst = accuracy_score(target_labels_tst, preds_labels_tst)

    # print('LSTM_FCN: ', acc_lstm)
    # print('InceptionTime: ', acc_cnn)
    # print('TST: ', acc_tst)
