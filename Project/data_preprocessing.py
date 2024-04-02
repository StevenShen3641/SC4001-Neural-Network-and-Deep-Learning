import logging
import os
import pickle

import gensim.downloader as api
import nltk
import numpy as np
import torch
import torch.nn.functional as F
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


def onehot(sentences, dimension):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    wvs = []
    for s in sentences:
        s = s.lower()
        indices = [alphabet.find(char) for char in s]
        indices = [len(alphabet) - 1 if i == -1 else i for i in indices]
        if len(indices) != dimension:
            indices += [len(alphabet) - 1] * (dimension - len(indices))
        indices = torch.tensor(indices)
        tensor = F.one_hot(indices, num_classes=len(alphabet)).float()
        tensor = tensor[:, :-1]
        wvs.append(tensor)
    wvs = torch.stack(wvs)
    wvs = torch.transpose(wvs, 1, 2)
    return wvs


def vectorize_sst2(model_type='glove-wiki-gigaword-200'):
    train_data, train_labels, test_data, test_labels, dev_data, dev_labels = load_sst()
    try:
        model = api.load(model_type)
        dimension = int(model_type.split('-')[-1])
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        exit(1)
    for p in wv_path:
        if not os.path.exists(p + f'_{model_type}_sst2.pkl'):
            for path in wv_path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            train_wv = vectorize(train_data, model, dimension)
            test_wv = vectorize(test_data, model, dimension)
            dev_wv = vectorize(dev_data, model, dimension)
            with open(wv_path[0] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([train_wv, train_labels], f)
            with open(wv_path[1] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([test_wv, test_labels], f)
            with open(wv_path[2] + f'_{model_type}_sst2.pkl', 'wb') as f:
                pickle.dump([dev_wv, dev_labels], f)
            break
    else:
        logging.info('Vector pickle files already exists!')


def onehot_sst2():
    sentences = ['ab cd']
    train_data, train_labels, test_data, test_labels, dev_data, dev_labels = load_sst()
    max_length = max(len(i) for i in train_data + test_data + dev_data)
    for p in wv_path:
        if not os.path.exists(p + f'_onehot_sst2.pkl'):
            for path in wv_path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            train_wv = onehot(train_data, dimension=max_length)
            test_wv = onehot(test_data, dimension=max_length)
            dev_wv = onehot(dev_data, dimension=max_length)

            with open(wv_path[0] + f'_onehot_sst2.pkl', 'wb') as f:
                pickle.dump([train_wv, train_labels], f)
            with open(wv_path[1] + f'_onehot_sst2.pkl', 'wb') as f:
                pickle.dump([test_wv, test_labels], f)
            with open(wv_path[2] + f'_onehot_sst2.pkl', 'wb') as f:
                pickle.dump([dev_wv, dev_labels], f)
            break
    else:
        logging.info('Onehot pickle files already exists!')


def main():
    # model_type = 'word2vec-google-news-300'
    # vectorize_sst2(model_type)
    onehot_sst2()


if __name__ == '__main__':
    main()
