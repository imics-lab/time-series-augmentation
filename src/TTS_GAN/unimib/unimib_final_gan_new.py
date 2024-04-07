from GANModels import *
import torch
import numpy as np
import os


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
    print("Downloading and processing UniMiB SHAR dataset, ADL Portion")
    x_train, y_train, x_validation, y_validation, x_test, y_test = unimib_load_dataset()
    print("\nUniMiB SHAR returned arrays:")
    print("x_train shape ",x_train.shape," y_train shape ", y_train.shape)
    print("x_valid shape ",x_validation.shape, "y_validation shape ", y_validation.shape)
    print("x_test shape  ",x_test.shape," y_test shape  ",y_test.shape)

    # x_train_total = np.concatenate((x_train, x_validation), axis = 0)
    # y_train_total = np.concatenate((y_train, y_validation), axis = 0)
    # print(x_train_total.shape, y_train_total.shape)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    path = '../logs_unimib'
    file_list = os.listdir(path)

    series_length = 151
    lstm_file = 'lstm_file.txt'
    cnn_file = 'cnn_file.txt'
    tst_file = 'tst_file.txt'
    lstm_acc_all = []
    cnn_acc_all = []
    tst_acc_all = []
    for shuffle_id in range(10):
        x_train_new = []
        y_train_new = []
        for file in file_list:
            parts = file.split('_')
            class_part = next(part for part in parts if 'class' in part)
            value = class_part.replace('class', '')
            class_id = int(value)
            model_id_path = '../logs_unimib/'+file+'/Model/checkpoint'
            x_train_id, y_train_id = extract_one_class(class_id = class_id)
    
            sample_size = x_train_id.shape[0]
            gen_net_id = Generator_z(seq_len=151, channels=4, latent_dim =100)
            ckp_id = torch.load(model_id_path)
            gen_net_id.load_state_dict(ckp_id['gen_state_dict'])
            
            z = torch.FloatTensor(np.random.normal(0, 1, (sample_size, 100)))
            x_train_id_new = gen_net_id(z)
            x_train_id_new = x_train_id_new.detach().numpy()
            y_train_id_new = [y_train_id[0]]*sample_size
            y_train_id_new = np.array(y_train_id_new)
            print(x_train_id.shape, y_train_id.shape)
            print(x_train_id_new.shape, y_train_id_new.shape)
            x_train_id_new = x_train_id_new.reshape(x_train_id_new.shape[0], series_length, x_train_id_new.shape[1])
            
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
        np.save('x_validation', x_validation)
        np.save('y_validation', y_validation)
        np.save('x_test', x_test)
        np.save('y_test', y_test)
        
    
        learn_lstm = train_model_lstm(x_train_new, y_train_new, x_validation, y_validation)
        preds_labels_lstm, target_labels_lstm = run_model(x_test, y_test, learn_lstm)
        acc_lstm = accuracy_score(target_labels_lstm, preds_labels_lstm)
        lstm_acc_all.append(acc_lstm)
        file = open(lstm_file,'a')
        write_str = str(acc_lstm)+ '--------'
        file.write(write_str)
        file.close()
    
        learn_cnn = train_model_cnn(x_train_new, y_train_new, x_validation, y_validation)
        preds_labels_cnn, target_labels_cnn = run_model(x_test, y_test, learn_cnn)
        acc_cnn = accuracy_score(target_labels_cnn, preds_labels_cnn)
        cnn_acc_all.append(acc_cnn)
        file = open(cnn_file,'a')
        write_str = str(acc_cnn)+ '--------'
        file.write(write_str)
        file.close()
    
        learn_tst = train_model_tst(x_train_new, y_train_new, x_validation, y_validation)
        preds_labels_tst, target_labels_tst = run_model(x_test, y_test, learn_tst)
        acc_tst = accuracy_score(target_labels_tst, preds_labels_tst)
        tst_acc_all.append(acc_tst)
        file = open(tst_file,'a')
        write_str = str(acc_tst)+ '--------'
        file.write(write_str)
        file.close()
    
    print('LSTM_FCN: ', lstm_acc_all)
    print('InceptionTime: ', cnn_acc_all)
    print('TST: ', tst_acc_all)


