import numpy as np
import pandas as pd
import csv
from tabulate import tabulate
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from AugmentTS.augmentts.augmenters.vae import LSTMVAE, VAEAugmenter

from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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
    train_file = 'train_mitbih_new.csv'
    test_file = 'test_mitbih_new.csv'
    data_train = pd.read_csv(train_file, header=0)
    data_test = pd.read_csv(test_file, header=0)
    print(data_train.shape)
    print(data_test.shape)
    
    x_train = data_train.iloc[:, :-1].values
    y_train = data_train.iloc[:, -1].values
    x_test = data_test.iloc[:, :-1].values
    y_test = data_test.iloc[:, -1].values

    y_train = to_categorical(y_train, num_classes=5)
    y_test = to_categorical(y_test, num_classes=5)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    x_train, x_valid, y_train, y_valid = train_test_split(\
            x_train, y_train, test_size = 0.25, shuffle = True)
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    series_length = 187
    n_channels = 1
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

    learn_lstm = train_model_lstm(x_train_new, y_train_new, x_valid, y_valid)
    preds_labels_lstm, target_labels_lstm = run_model(x_test, y_test, learn_lstm)
    acc_lstm = accuracy_score(target_labels_lstm, preds_labels_lstm)

    learn_cnn = train_model_cnn(x_train_new, y_train_new, x_valid, y_valid)
    preds_labels_cnn, target_labels_cnn = run_model(x_test, y_test, learn_cnn)
    acc_cnn = accuracy_score(target_labels_cnn, preds_labels_cnn)

    learn_tst = train_model_tst(x_train_new, y_train_new, x_valid, y_valid)
    preds_labels_tst, target_labels_tst = run_model(x_test, y_test, learn_tst)
    acc_tst = accuracy_score(target_labels_tst, preds_labels_tst)

    print('LSTM_FCN: ', acc_lstm)
    print('InceptionTime: ', acc_cnn)
    print('TST: ', acc_tst)