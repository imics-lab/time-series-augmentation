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


class eeg_allClass():
    def __init__(self):
        input_dir = '../eeg_dataset_1/'
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
        
        self.x_train = self.x_train[:, :128,:]
  
        self.x_train = self.x_train.reshape\
        (self.x_train.shape[0], self.x_train.shape[2], self.x_train.shape[1])

        self.y_train = np.argmax(self.y_train, axis=1) #deencode

        print(self.x_train.shape, self.y_train.shape)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

def main():
    class_all = eeg_allClass() #mitbih_oneClass(class_id = 0)
            
if __name__ == "__main__":
    main()