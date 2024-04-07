from sklearn.utils import shuffle
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding

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


if __name__ == "__main__":
    x_train_new = np.load('x_train_vae_unimib.npy')
    y_train_new = np.load('y_train_vae_unimib.npy')
    print(x_train_new.shape, y_train_new.shape)
    x_validation = np.load('x_validation.npy')
    y_validation = np.load('y_validation.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    filename ='unimib_vae.txt'

    Z=1.96
    acc_list_mean = []
    for shuffle_id in range(10):
        x_train_shuffled, y_train_shuffled = shuffle(x_train_new, y_train_new)
        learn = train_model_lstm(x_train_shuffled, y_train_shuffled, x_validation, y_validation)
        preds_labels, target_labels = run_model(x_test, y_test, learn)
        acc_list_mean.append(accuracy_score(target_labels, preds_labels))
    
    mean_accuracy_lstm = np.mean(acc_list_mean)
    std_dev = np.std(acc_list_mean, ddof=1)  # ddof=1 for sample standard deviation
    std_error = std_dev / np.sqrt(len(acc_list_mean))
    interval_range_lstm = Z*std_error  

    file = open(filename,'a')
    write_str = str(mean_accuracy_lstm)+'  '+str(interval_range_lstm)+ ' -------- '
    file.write(write_str)
    file.close()
    

    acc_list_mean = []
    for shuffle_id in range(10):
        x_train_shuffled, y_train_shuffled = shuffle(x_train_new, y_train_new)
        learn = train_model_cnn(x_train_shuffled, y_train_shuffled, x_validation, y_validation)
        preds_labels, target_labels = run_model(x_test, y_test, learn)
        acc_list_mean.append(accuracy_score(target_labels, preds_labels))
    
    mean_accuracy_cnn = np.mean(acc_list_mean)
    std_dev = np.std(acc_list_mean, ddof=1)  # ddof=1 for sample standard deviation
    std_error = std_dev / np.sqrt(len(acc_list_mean))
    interval_range_cnn = Z*std_error

    file = open(filename,'a')
    write_str = str(mean_accuracy_cnn)+'  '+str(interval_range_cnn)+ ' -------- '
    file.write(write_str)
    file.close()

    Z=1.96
    acc_list_mean = []
    for shuffle_id in range(10):
        x_train_shuffled, y_train_shuffled = shuffle(x_train_new, y_train_new)
        learn = train_model_tst(x_train_shuffled, y_train_shuffled, x_validation, y_validation)
        preds_labels, target_labels = run_model(x_test, y_test, learn)
        acc_list_mean.append(accuracy_score(target_labels, preds_labels))
    
    mean_accuracy_tst = np.mean(acc_list_mean)
    std_dev = np.std(acc_list_mean, ddof=1)  # ddof=1 for sample standard deviation
    std_error = std_dev / np.sqrt(len(acc_list_mean))
    interval_range_tst = Z*std_error
    
    file = open(filename,'a')
    write_str = str(mean_accuracy_tst)+'  '+str(interval_range_tst)+ ' -------- '
    file.write(write_str)
    file.close()

    print('LSTM')
    print(mean_accuracy_lstm)
    print(interval_range_lstm)

    print('CNN')
    print(mean_accuracy_cnn)
    print(interval_range_cnn)

    print('TST')
    print(mean_accuracy_tst)
    print(interval_range_tst)