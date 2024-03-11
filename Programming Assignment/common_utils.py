### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class MLP(nn.Module):

    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            # model layers
            # here we add 0.2 dropout rate to avoid overfitting problem

            # first layer with ReLU activation function
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),  # dropout of probability 0.2

            # second layer with ReLU activation function
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),  # dropout of probability 0.2

            # third layer with ReLU activation function
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),  # dropout of probability 0.2

            # output layer with Sigmoid activation function
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()

        )

    # YOUR CODE HERE
    # define the forward pass
    def forward(self, x):
        # pass through the self.mlp_stack
        logits = self.mlp_stack(x)
        return logits


class CustomDataset(Dataset):
    # YOUR CODE HERE
    def __init__(self, X, y):
    # convert data to tensors
    # note that long type is needed for y to prevent exception
        self.X =torch.tensor(X, dtype=torch.float)
        self.y =torch.tensor(y, dtype=torch.long)

    # return size of data
    def __len__(self):
        return len(self.y)

    # return data with given index
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    

loss_fn = nn.BCELoss()
