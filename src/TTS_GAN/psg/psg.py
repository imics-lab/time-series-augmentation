# -*- coding: utf-8 -*-

from sklearn.utils import shuffle
from tsai.models.utils import *
from tsai.basics import *
from tsai.inference import load_learner
from tsai.all import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding

# def load_psg_data():
#     input_dir = 'PSG/'
#     x_train = np.load(input_dir +'x_train.npy')
#     x_validation = np.load(input_dir +'x_valid.npy')
#     x_test = np.load(input_dir +'x_test.npy')
#     y_train = np.load(input_dir +'y_train.npy')
#     y_validation = np.load(input_dir +'y_valid.npy')
#     y_test = np.load(input_dir +'y_test.npy')
    
#     return x_train, y_train, x_validation, y_validation, x_test, y_test

class psg_oneClass():
    def __init__(self, class_id = 1):
        self.class_id = class_id
        input_dir = '../Medical_Data/Sleep/PSG/'
        self.x_train = np.load(input_dir +'x_train.npy')
        
        #self.x_validation = np.load(input_dir +'x_valid.npy')
        #self.x_test = np.load(input_dir +'x_test.npy')
        #self.y_validation = np.load(input_dir +'y_valid.npy')
        #self.y_test = np.load(input_dir +'y_test.npy')
        
        self.y_train = np.load(input_dir +'y_train.npy')
        
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
    class_all = psg_oneClass() #mitbih_oneClass(class_id = 0)
            
if __name__ == "__main__":
    main()