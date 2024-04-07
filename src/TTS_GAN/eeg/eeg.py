from sklearn.utils import shuffle
# from tsai.models.utils import *
# from tsai.basics import *
# from tsai.inference import load_learner
# from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding

# def load_eeg_data():
#     input_dir = '../Medical_Data/eeg_dataset_1/'
#     x_train = np.load(input_dir +'x_train.npy')
#     x_validation = np.load(input_dir +'x_valid.npy')
#     x_test = np.load(input_dir +'x_test.npy')
#     y_train = np.load(input_dir +'y_train.npy')
#     y_validation = np.load(input_dir +'y_valid.npy')
#     y_test = np.load(input_dir +'y_test.npy')
    
#     x_train = x_train.astype(float)
#     x_validation = x_validation.astype(float)
#     x_test = x_test.astype(float)
    
#     return x_train, y_train, x_validation, y_validation, x_test, y_test


class eeg_oneClass():
    def __init__(self, class_id = 1):
        self.class_id = class_id
        input_dir = '../Medical_Data/EEG_Data/eeg_dataset_1/'
        
        self.x_train = np.load(input_dir +'x_train.npy')
        self.y_train = np.load(input_dir +'y_train.npy')    
        self.x_train = self.x_train.astype(float)

        new_x =[]
        new_y = []
        split_size = 178
        n_splits = 23
        for i in range(self.x_train.shape[0]):
            sub_data = self.x_train[i]
            y_val = self.y_train[i]
            for i in range (n_splits):
                sample = sub_data[i*split_size:(i+1)*split_size]
                new_x.append(sample)
                temp_y = y_val
                new_y.append(temp_y)
                #print(sample.shape)
                
        new_x = np.array(new_x)
        new_y = np.array(new_y)
        self.x_train = new_x
        self.y_train = new_y

        
        self.one_class_train_data, self.one_class_train_labels = \
            self.extract_one_class(self.class_id, self.x_train, self.y_train)

        print(self.x_train.shape, self.y_train.shape)
        print(self.one_class_train_data.shape)
        self.one_class_train_data = self.one_class_train_data.reshape\
        (self.one_class_train_data.shape[0], self.one_class_train_data.shape[2], 1, self.one_class_train_data.shape[1])
    
    def extract_one_class(self, class_id, x_train, y_train):
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
        
    def __len__(self):
        return len(self.one_class_train_labels)
    
    def __getitem__(self, idx):
        return self.one_class_train_data[idx], self.one_class_train_labels[idx]

def main():
    class_all = eeg_oneClass() 
            
if __name__ == "__main__":
    main()