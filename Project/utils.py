import os
import random
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

crowdflower_path = "./data/text_emotion.csv"
wassa2017_path = "./data/wassa2017/training/anger-ratings-0to1.train.txt"
wassa2017_dir = "./data/wassa2017/"
train_data_path = "./data/train_data/train.csv"
test_data_path = "./data/test_data/test.csv"

def split(df, test_size=0.2):
    train_data = {}
    test_data = {}
    for s in df['sentiment'].unique():
        data = df[df['sentiment'] == s]

        train_df, test_df = train_test_split(
            data, test_size=test_size, random_state=42)
        train_data[s] = train_df
        test_data[s] = test_df

    train_data = pd.concat(train_data.values(), ignore_index=True)
    test_data = pd.concat(test_data.values(), ignore_index=True)
    if not os.path.exists(train_data_path):
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    if not os.path.exists(test_data_path):
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    X_train = train_data['content'].tolist()
    y_train = train_data['sentiment'].tolist()
    X_test = test_data['content'].tolist()
    y_test = test_data['sentiment'].tolist()

    return X_train, X_test, y_train, y_test


def load_wassa():
    if not os.path.exists(wassa2017_dir):
        logging.warning('Wassa does not exist!!')
        exit(1)

    else:
        wassa_train_X = []
        wassa_train_y = []
        wassa_test_X = []
        wassa_test_y = []
        train_path = "training/"
        test_path = "testing/"
        for j in ["anger", "fear", "joy", "sadness"]:
            with open(wassa2017_dir + train_path + j + "-ratings-0to1.train.txt", "r", encoding="utf-8") as file:
                file.readline()
                for line in file:
                    columns = line.strip().split("\t")
                    wassa_train_X.append(columns[1])
                    wassa_train_y.append(j)
            with open(wassa2017_dir + test_path + j + "-ratings-0to1.test.target.txt", "r", encoding="utf-8") as file:
                file.readline()
                for line in file:
                    columns = line.strip().split("\t")
                    wassa_test_X.append(columns[1])
                    wassa_test_y.append(j)

    wassa_train_df = pd.DataFrame({'sentiment': wassa_train_y,
                                   'content': wassa_train_X})
    wassa_test_df = pd.DataFrame({'sentiment': wassa_test_y,
                                  'content': wassa_test_X})
    return wassa_train_df, wassa_test_df
