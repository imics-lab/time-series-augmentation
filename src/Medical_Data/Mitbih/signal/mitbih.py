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


class mitbih_allClass():
    def __init__(self):
        train_file = '../train_mitbih_new.csv'
        #test_file = '../test_mitbih_new.csv'
        data_train = pd.read_csv(train_file, header=0)
        #data_test = pd.read_csv(test_file, header=0)
        print(data_train.shape)
        #print(data_test.shape)
        
        self.x_train = data_train.iloc[:, :-1].values
        self.y_train = data_train.iloc[:, -1].values
        # x_test = data_test.iloc[:, :-1].values
        # y_test = data_test.iloc[:, -1].values
    
        #self.y_train = to_categorical(self.y_train, num_classes=5)
        #y_test = to_categorical(y_test, num_classes=5)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        self.x_train = self.x_train[:, :128,:]
        self.x_train = self.x_train.reshape\
        (self.x_train.shape[0], self.x_train.shape[2], self.x_train.shape[1])
    
        self.x_train, x_valid, self.y_train, y_valid = train_test_split(\
                self.x_train, self.y_train, test_size = 0.25, shuffle = True, random_state = 123)
        #self.y_train = np.argmax(self.y_train_one_hot, axis=1) #deencode

        print(self.x_train.shape, self.y_train.shape)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

def main():
    class_all = mitbih_allClass() #mitbih_oneClass(class_id = 0)
            
if __name__ == "__main__":
    main()