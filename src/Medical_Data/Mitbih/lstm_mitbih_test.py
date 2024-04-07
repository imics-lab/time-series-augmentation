import numpy as np
import pandas as pd
import csv
from tabulate import tabulate
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def train_model(x_train, y_train, x_valid, y_valid):
    X, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms  = [None, TSClassification()] # TSClassification == Categorize
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 32])
    #dls.dataset
    model = build_ts_model(LSTM_FCNPlus, dls=dls)
    learn = Learner(dls, model, metrics=accuracy)
    learn.fit_one_cycle(10, lr_max=1e-2)
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
    
    x_train_total = x_train.copy()
    y_train_total = y_train.copy()

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    filename = 'lstm.txt'
    print('JITTERING')
    jitter_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 0.8, 1]
    jitter_opt = testing_values(x_train_total, y_train_total, jitter_values, 'Jitter')
    print(jitter_opt)
    file = open(filename,'a')
    file.write('Jitter'+str(jitter_opt))
    file.close()
    
    print('SCALING')
    scale_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    scale_opt = testing_values(x_train_total, y_train_total, scale_values, 'Scale')
    print(scale_opt)
    file = open(filename,'a')
    file.write('Scale'+str(scale_opt))
    file.close()

    print('MAGNITUDE WARPING')
    mag_warp_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    mag_warp_opt = testing_values(x_train_total, y_train_total, mag_warp_values, 'MagWarp')
    print(mag_warp_opt)
    file = open(filename,'a')
    file.write('Mag Warp'+str(mag_warp_opt))
    file.close()

    print('TIME WARPING')
    time_warp_values = [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 9, 11]
    time_warp_opt = testing_values(x_train_total, y_train_total, time_warp_values, 'TimeWarp')
    print(time_warp_opt)
    file = open(filename,'a')
    file.write('Time Warp'+str(time_warp_opt))
    file.close()
    
    print('WINDOW WARPING')
    window_warp_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]
    window_warp_opt = testing_values(x_train_total, y_train_total, window_warp_values, 'WinWarp')
    print(window_warp_opt)
    file = open(filename,'a')
    file.write('Window Warp'+str(window_warp_opt))
    file.close()
    
    print('WINDOW SLICING')
    window_slice_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    window_slice_opt = testing_values(x_train_total, y_train_total, window_slice_values, 'WinSlice')
    print(window_slice_opt)
    file = open(filename,'a')
    file.write('Win Slice Warp'+str(window_slice_opt))
    file.close()
    
    print('LSTM_FCN')
    final_values_dict = {'Jitter': jitter_opt,\
                'Scale': scale_opt,\
                'Magnitude Warp': mag_warp_opt,\
                'Time Warp': time_warp_opt,\
                'Window Warp': window_warp_opt,\
                'Window Slicing': window_slice_opt}