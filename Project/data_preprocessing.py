import logging
import os
import pickle
import gc

import gensim.downloader as api
import nltk
import numpy as np
import torch
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
wv_path = ['./data/train_data/train', './data/test_data/test', './data/dev_data/dev']


def load_sst():
    sst2 = load_dataset('glue', 'sst2')
    logging.info("SST-2 loaded")
    train_data = sst2["train"]["sentence"]
    train_labels = sst2["train"]["label"]
    test_data = sst2["test"]["sentence"]
    test_labels = sst2["test"]["label"]
    dev_data = sst2["validation"]["sentence"]
    dev_labels = sst2["validation"]["label"]
    return train_data, train_labels, test_data, test_labels, dev_data, dev_labels


def vectorize(sentences, model, dimension, wv_type='zero_padding'):
    tokens = [nltk.word_tokenize(s.lower()) for s in sentences]


    wvs = []
    for t in tokens:
        wv = []
        for w in t:
            if wv_type == 'zero_padding':
                try:
                    wv.append(model[w])
                except KeyError:
                    wv.append(np.zeros(dimension))
        wvs.append(wv)

    return wvs


def process_sst2(model_type='glove-wiki-gigaword-200'):
    train_data, train_labels, test_data, test_labels, dev_data, dev_labels = load_sst()
    try:
        model = api.load(model_type)
        dimension = int(model_type.split('-')[-1])
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        exit(1)
    for p in wv_path:
        if not os.path.exists(p + f'_{model_type}_sst2.pkl'):
            for p in wv_path:
                os.makedirs(os.path.dirname(p), exist_ok=True)
            train_wv = vectorize(train_data, model, dimension)
            test_wv = vectorize(test_data, model, dimension)
            dev_wv = vectorize(dev_data, model, dimension)
            max_length = max(len(i) for i in train_wv + test_wv + dev_wv)
            del model
            del train_data, test_data, dev_data
            gc.collect()
            print(max_length)
            train_tensor = convert_to_tensor(train_wv, max_length)
            test_tensor = convert_to_tensor(train_wv, max_length)
            dev_tensor = convert_to_tensor(train_wv, max_length)

            with open(wv_path[0] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([train_tensor, train_labels], f)
            with open(wv_path[1] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([test_tensor, test_labels], f)
            with open(wv_path[2] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([dev_tensor, dev_labels], f)
            break
    else:
        logging.info('Vector pickle files already exists!')


def convert_to_tensor(X, max_length):
    wv_num = X[0][0].shape[0]
    X_list = []
    for i in X:
        temp = i
        for _ in range(max_length - len(i)):
            temp.append(np.zeros(wv_num))
        np_array = np.array(temp)
        trans = np.transpose(np_array)
        X_list.append(torch.from_numpy(trans))
    X_tensor = torch.stack(X_list)
    return X_tensor


def main():
    model_type = 'word2vec-google-news-300'
    process_sst2(model_type)


if __name__ == '__main__':
    main()
