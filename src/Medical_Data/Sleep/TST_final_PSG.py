import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.utils import shuffle
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def load_psg_data():
    input_dir = 'PSG/'
    x_train = np.load(input_dir +'x_train.npy')
    x_validation = np.load(input_dir +'x_valid.npy')
    x_test = np.load(input_dir +'x_test.npy')
    y_train = np.load(input_dir +'y_train.npy')
    y_validation = np.load(input_dir +'y_valid.npy')
    y_test = np.load(input_dir +'y_test.npy')
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def train_model(x_train, y_train, x_valid, y_valid):
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

def final_values(x_train_total, y_train_total, augment_value, augment_func):
    acc_list_mean = []
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
            x_train_augment_initial = window_warp(x_train_new, window_warp_val, 4)
        elif (augment_func == 'WinSlice'):
            x_train_augment_initial = window_slice(x_train_new, augment_value)
            
        x_train_augment = np.concatenate((x_train_new, x_train_augment_initial), axis = 0)
        y_train_augment = np.concatenate((y_train_new, y_train_new), axis = 0)
        x_train_shuffled, y_train_shuffled = shuffle(x_train_augment, y_train_augment)
        learn = train_model(x_train_shuffled, y_train_shuffled, x_validate, y_validate)
        preds_labels, target_labels = run_model(x_test, y_test, learn)
        acc_list_mean.append(accuracy_score(target_labels, preds_labels))
    
    mean_accuracy = np.mean(acc_list_mean)
    std_dev = np.std(acc_list_mean, ddof=1)  # ddof=1 for sample standard deviation
    std_error = std_dev / np.sqrt(len(acc_list_mean))
    interval_range = Z*std_error
    
    print(augment_func)
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    
    augment_dict = {'Name': augment_func,\
                     'Mean Accuracy': mean_accuracy,\
                     'Std Deviation': std_dev,\
                     'Interval Range': interval_range}
    return augment_dict

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
    x_train_total = np.concatenate((x_train, x_valid), axis = 0)
    y_train_total = np.concatenate((y_train, y_valid), axis = 0)
    print(x_train_total.shape, y_train_total.shape)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    Z=1.96
    acc_list_original = []
    for shuffle_id in range(10):
        x_train_new, x_validate, y_train_new, y_validate = train_test_split(\
            x_train_total, y_train_total, test_size = 0.25, shuffle = True)
        x_train_shuffled, y_train_shuffled = shuffle(x_train_new, y_train_new)

        learn = train_model(x_train_shuffled, y_train_shuffled, x_validate, y_validate)
        preds_labels, target_labels = run_model(x_test, y_test, learn)
    
        # Print print prediction accuracy
        print('Prediction accuracy: {0:.3f}'.format(accuracy_score(target_labels, preds_labels)))
        acc_list_original.append(accuracy_score(target_labels, preds_labels))
    

    # Calculate the mean accuracy and population standard deviation
    mean_accuracy = np.mean(acc_list_original)
    std_dev = np.std(acc_list_original, ddof=1)  # ddof=1 for sample standard deviation
    std_error = std_dev / np.sqrt(len(acc_list_original))
    
    # Calculate the confidence interval
    conf_interval_lower = mean_accuracy - Z * std_error
    conf_interval_upper = mean_accuracy + Z * std_error
    print("Original")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Z-score: {Z:.4f}")
    print(f"Confidence Interval (95%): ({conf_interval_lower:.4f}, {conf_interval_upper:.4f})")
    interval_range = Z*std_error
    original_dict = {'Name': 'Original',\
                     'Mean Accuracy': mean_accuracy,\
                     'Std Deviation': std_dev,\
                     'Interval Range': interval_range}

    print('JITTERING')
    jitter_val = 0.07
    jitter_dict = final_values(x_train_total, y_train_total, jitter_val, 'Jitter')
    
    print('SCALING')
    scale_val = 9
    scale_dict = final_values(x_train_total, y_train_total, scale_val, 'Scale')

    print('MAGNITUDE WARPING')
    mag_warp_val = 9
    mag_warp_dict = final_values(x_train_total, y_train_total, mag_warp_val, 'MagWarp')

    print('TIME WARPING')
    time_warp_val = 0.1
    time_warp_dict = final_values(x_train_total, y_train_total, time_warp_val, 'TimeWarp')
    
    print('WINDOW WARPING')
    window_warp_val = 0.09
    window_warp_dict = final_values(x_train_total, y_train_total, window_warp_val, 'WinWarp')
    
    print('WINDOW SLICING')
    window_slice_val = 0.7
    window_slice_dict = final_values(x_train_total, y_train_total, window_slice_val, 'WinSlice')
    
    print('TST')
    print(original_dict)
    print(jitter_dict)
    print(scale_dict)
    print(mag_warp_dict)
    print(time_warp_dict)
    print(window_warp_dict)
    print(window_slice_dict)
    