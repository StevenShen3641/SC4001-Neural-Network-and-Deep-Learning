import nltk
import logging
import pickle
import os
import numpy as np
import pandas as pd
import gensim.downloader as api
from utils import train_data_path, test_data_path, load_wassa, split


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# unknown_words = {}
wv_path = ['./data/train_data/train', './data/test_data/test']


def vectorize(sentences, model_type='glove-twitter-25', wv_type='zero_padding'):
    tokens = [nltk.word_tokenize(s.lower()) for s in sentences]
    try:
        model = api.load(model_type)
        dimension = int(model_type.split('-')[-1])
    except Exception as e:
        logging.warning(f"An error occurred: {e}")
        exit(1)

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


def main():
    model_type = 'glove-twitter-200'  # 25 100 200 word2vec 300 
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        train_df = pd.read_csv(train_data_path)
        X_train, y_train = train_df['content'].tolist(
        ), train_df['sentiment'].tolist()
        test_df = pd.read_csv(test_data_path)
        X_test, y_test = test_df['content'].tolist(
        ), test_df['sentiment'].tolist()
    else:
        train_data, test_data = load_wassa()
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        X_train = train_data['content'].tolist()
        y_train = train_data['sentiment'].tolist()
        X_test = test_data['content'].tolist()
        y_test = test_data['sentiment'].tolist()

    for p in wv_path:
        if not os.path.exists(p):
            train_wv = vectorize(X_train, model_type=model_type)
            test_wv = vectorize(X_test, model_type=model_type)
            with open(wv_path[0] + f'_{model_type}.pkl', 'wb') as f:
                pickle.dump([train_wv, y_train], f)
            with open(wv_path[1] + f'_{model_type}.pkl', 'wb') as f:
                pickle.dump([test_wv, y_test], f)
            break
    else:
        logging.info('Vector pickle files already exists!')


if __name__ == '__main__':
    main()
