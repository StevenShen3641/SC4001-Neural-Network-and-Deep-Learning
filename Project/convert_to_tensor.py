import logging
import pickle

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
wv_path = ['./data/train_data/train', './data/test_data/test', './data/dev_data/dev']


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
    with open(wv_path[0] + f'_{model_type}_sst2.pkl', 'rb') as f:
        train_wv, train_labels = pickle.load(f)
    with open(wv_path[1] + f'_{model_type}_sst2.pkl', 'rb') as f:
        test_wv, test_labels = pickle.load(f)
    with open(wv_path[2] + f'_{model_type}_sst2.pkl', 'rb') as f:
        dev_wv, dev_labels = pickle.load(f)
    max_length = max(len(i) for i in train_wv + test_wv + dev_wv)

    train_tensor = convert_to_tensor(train_wv, max_length)
    torch.save(train_tensor, wv_path[0] + f'_{model_type}_tensor_sst2.pt')
    logging.info("train done!")
    test_tensor = convert_to_tensor(test_wv, max_length)
    torch.save(test_tensor, wv_path[1] + f'_{model_type}_tensor_sst2.pt')
    logging.info("test done!")
    dev_tensor = convert_to_tensor(dev_wv, max_length)
    torch.save(dev_tensor, wv_path[2] + f'_{model_type}_tensor_sst2.pt')
    logging.info("dev done!")
    with open(wv_path[0] + f'_{model_type}_label_sst2.pkl', 'wb') as f:
        pickle.dump(train_labels, f)
    with open(wv_path[1] + f'_{model_type}_label_sst2.pkl', 'wb') as f:
        pickle.dump(test_labels, f)
    with open(wv_path[2] + f'_{model_type}_label_sst2.pkl', 'wb') as f:
        pickle.dump(dev_labels, f)


if __name__ == '__main__':
    main()
