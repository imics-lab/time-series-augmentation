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

class psg_allClass():
    def __init__(self):
        input_dir = 'PSG/'
        self.x_train = np.load(input_dir +'x_train.npy')
        #self.x_validation = np.load(input_dir +'x_valid.npy')
        #self.x_test = np.load(input_dir +'x_test.npy')
        y_train_one_hot = np.load(input_dir +'y_train.npy')
        #self.y_validation = np.load(input_dir +'y_valid.npy')
        #self.y_test = np.load(input_dir +'y_test.npy')
        self.y_train = np.argmax(y_train_one_hot, axis=1) #deencode

        pad_width = [(0, 0), (0, 512 - self.x_train.shape[1]), (0, 0)]
        # Pad the array with zeros
        self.x_train = np.pad(self.x_train, pad_width, mode='constant', constant_values=0)
        #self.x_train = self.x_train[:,:128,:]
        print(self.x_train.shape, self.y_train.shape)
        self.x_train = self.x_train.reshape\
        (self.x_train.shape[0], self.x_train.shape[2], self.x_train.shape[1])
        print(self.x_train.shape, self.y_train.shape)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

def main():
    class_all = psg_allClass() #mitbih_oneClass(class_id = 0)
            
if __name__ == "__main__":
    main()