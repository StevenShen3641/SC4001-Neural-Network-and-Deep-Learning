from utils import load_data, split
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors, Word2Vec
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(format='%(levelname)s: %(message)s')
types = ['twitter', 'google', 'none']
unknown_words = {}


def vectorize(sentences, type='glove-twitter-25'):
    dimension = 0
    tokens = [nltk.word_tokenize(s.lower()) for s in sentences]
    try:
        model = api.load(type)
        dimension = type.split('-')[-1]
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        exit(1)

    wvs = []
    print(model['is'])
    # for s in sentences:
    #     wv = []
    #     for w in s:
    #         try:
    #             wv.append(model[w])
    #         except KeyError:
    #             try:
    #                 wv.append(unknown_words[w])
    #             except KeyError:
    #                 unknown_words[w] = np.random.uniform(-0.25, 0.25, dimension)

    return 0


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    train_wv = vectorize(X_train)


if __name__ == '__main__':
    main()
