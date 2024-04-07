# -*- coding: utf-8 -*-

from sklearn.utils import shuffle
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from AugmentTS.augmentts.augmenters.vae import LSTMVAE, VAEAugmenter
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding

def load_psg_data():
    input_dir = 'PSG/'
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

    np.save('x_train_psg', x_train)
    np.save('y_train_psg', y_train)
    np.save('x_valid', x_valid)
    np.save('y_valid', y_valid)
    np.save('x_test', x_test)
    np.save('y_test', y_test)
    # x_train_total = np.concatenate((x_train, x_validation), axis = 0)
    # y_train_total = np.concatenate((y_train, y_validation), axis = 0)
    # print(x_train_total.shape, y_train_total.shape)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print("Available GPUs:", gpus)
    
    # # Set TensorFlow to only use the first GPU
    # if gpus:
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[5], 'GPU')
    #         tf.config.experimental.set_memory_growth(gpus[5], True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         print(e)
    series_length = 500
    n_channels = 12
    vae = LSTMVAE(series_len=series_length, n_channels = n_channels)
    #sample_size = 1000
    n_epochs = 150
    x_train_new = []
    y_train_new = []
    
    for class_id in range(y_train.shape[1]):
        x_train_id, y_train_id = extract_one_class(class_id)
        x_train_id = x_train_id.reshape(x_train_id.shape[0], x_train_id.shape[2], series_length)
        sample_size = x_train_id.shape[0]
        augmenter_id = VAEAugmenter(vae)
 
        augmenter_id.fit(x_train_id, epochs=n_epochs, batch_size=None)
        x_train_id_new = augmenter_id.sample(n=sample_size)
        y_train_id_new = [y_train_id[0]]*sample_size
        y_train_id_new = np.array(y_train_id_new)
        print(x_train_id_new.shape, y_train_id_new.shape)
        
        # x_train_augment = np.concatenate([x_train_id_new, x_train_new, x_train_id], axis=0)
        # y_train_augment = np.concatenate([y_train_id_new, y_train_new, y_train_id], axis=0)
        x_train_augment = np.concatenate([x_train_id_new, x_train_id], axis=0)
        y_train_augment = np.concatenate([y_train_id_new, y_train_id], axis=0)
        x_train_new.append(x_train_augment)
        y_train_new.append(y_train_augment)
        #print(x_train_new.shape, y_train_new.shape)

    x_train_new = np.concatenate(x_train_new, axis=0)
    y_train_new = np.concatenate(y_train_new, axis=0)
    x_train_new = x_train_new.reshape(x_train_new.shape[0], series_length, x_train_new.shape[1])
    print(x_train_new.shape, y_train_new.shape)
    x_train_new, y_train_new = shuffle(x_train_new, y_train_new, random_state=42)
    
    np.save('x_train_vae', x_train_new)
    np.save('y_train_vae', y_train_new)
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
