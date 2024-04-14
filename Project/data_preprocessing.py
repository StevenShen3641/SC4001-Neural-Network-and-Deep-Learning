import logging
import os
import pickle
import string
import nltk
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from utils import *

nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
wv_path = ['./data/train_data/train', './data/test_data/test', './data/dev_data/dev']


def load_sst():
    """
    load sst-2 dataset
    """
    sst2 = load_dataset('glue', 'sst2')
    logging.info("SST-2 loaded")
    train_data = sst2["train"]["sentence"]
    train_labels = sst2["train"]["label"]
    test_data = sst2["test"]["sentence"]
    test_labels = sst2["test"]["label"]
    dev_data = sst2["validation"]["sentence"]
    dev_labels = sst2["validation"]["label"]
    return train_data, train_labels, test_data, test_labels, dev_data, dev_labels


def load_IMDB():
    """
    load IMDB dataset
    """
    df = pd.read_csv('./data/IMDB/IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
    return df


def onehot(sentences, dimension):
    """
    onehot embeddings of given sentences
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    wvs = []
    for s in sentences:
        s = s.lower()
        # transfer characters to indices
        indices = [alphabet.find(char) for char in s][:dimension]
        indices = [len(alphabet) - 1 if i == -1 else i for i in indices]
        # add blank spaces
        if len(indices) != dimension:
            indices += [len(alphabet) - 1] * (dimension - len(indices))
        indices = torch.tensor(indices)
        tensor = F.one_hot(indices, num_classes=len(alphabet)).float()
        # replace the spaces with zero paddings
        tensor = tensor[:, :-1]
        wvs.append(tensor)
    wvs = torch.stack(wvs)
    wvs = torch.transpose(wvs, 1, 2)
    return wvs


def onehot_IMDB():
    """
    embed IMDB set with onehot tensors
    """
    df = load_IMDB()
    test_data = df['review'].to_list()
    test_labels = df['sentiment'].to_list()
    max_length = 268  # set max length
    p = wv_path[1]
    # create path first
    if not os.path.exists(p + f'_onehot_IMDB.pkl'):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        test_wv = onehot(test_data, dimension=max_length)
        # store file
        with open(wv_path[1] + f'_onehot_IMDB.pkl', 'wb') as f:
            pickle.dump([test_wv, test_labels], f)
    else:
        logging.info('Onehot pickle files already exists!')


def onehot_sst2():
    """
    embed sst-2 set with onehot tensors
    """
    # load set
    train_data, train_labels, test_data, test_labels, dev_data, dev_labels = load_sst()
    df_train = pd.DataFrame({'content': train_data, 'sentiment': train_labels})
    df_train.reset_index(drop=True, inplace=True)
    df_dev = pd.DataFrame({'content': dev_data, 'sentiment': dev_labels})
    df_dev.reset_index(drop=True, inplace=True)

    # remove stop words to create another set
    remove_special_content(df_train)
    remove_special_content(df_dev)
    punct = word_tokenize(string.punctuation) + ['``', '...', '..', '\'s', '--', '-', 'n\'t', '\'', '(', ')',
                                                 '[', ']', '{', '}']
    train_data_nonstop = preprocess(df_train['content']).map(lambda x: '. '.join(x)).map(
        lambda x: replace_punct(x, punct))
    dev_data_nonstop = preprocess(df_dev['content']).map(lambda x: '. '.join(x)).map(
        lambda x: replace_punct(x, punct))

    max_length = 268
    # store ordinary dataset
    for p in wv_path:
        if not os.path.exists(p + f'_onehot_sst2.pkl'):
            for path in wv_path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(wv_path[0] + f'_onehot_sst2.pkl', 'wb') as f:
                train_wv = onehot(train_data, dimension=max_length)
                pickle.dump([train_wv, train_labels], f)
            with open(wv_path[1] + f'_onehot_sst2.pkl', 'wb') as f:
                test_wv = onehot(test_data, dimension=max_length)
                pickle.dump([test_wv, test_labels], f)
            with open(wv_path[2] + f'_onehot_sst2.pkl', 'wb') as f:
                dev_wv = onehot(dev_data, dimension=max_length)
                pickle.dump([dev_wv, dev_labels], f)
            break
    else:
        logging.info('onehot_sst2 pickle files already exists!')
    # store non-stop dataset
    if not os.path.exists(wv_path[0] + f'_onehot_sst2_nonstop.pkl'):
        train_wv = onehot(train_data_nonstop, dimension=max_length)
        with open(wv_path[0] + f'_onehot_sst2_nonstop_nonstop.pkl', 'wb') as f:
            pickle.dump([train_wv, train_labels], f)
    elif not os.path.exists(wv_path[2] + f'_onehot_sst2_nonstop.pkl'):
        dev_wv = onehot(dev_data_nonstop, dimension=max_length)
        with open(wv_path[2] + f'_onehot_sst2_nonstop_nonstop.pkl', 'wb') as f:
            pickle.dump([dev_wv, dev_labels], f)
    else:
        logging.info('onehot_sst2_nonstop pickle files already exists!')


def main():
    onehot_sst2()
    onehot_IMDB()


if __name__ == '__main__':
    main()
