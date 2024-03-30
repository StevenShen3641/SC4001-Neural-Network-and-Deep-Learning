import os
import random
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

crowdflower_path = "./data/text_emotion.csv"
wassa2017_path = "./data/wassa2017/training/anger-ratings-0to1.train.txt"
train_data_path = "./data/train_data/train.csv"
test_data_path = "./data/test_data/test.csv"


def load_data(sample=True):
    if not os.path.exists(crowdflower_path):
        cf_df = pd.read_csv('https://query.data.world/s/x5wh4megwt4fd3jyotb75cz6qat33e?dws=00000')
        os.makedirs(os.path.dirname(crowdflower_path), exist_ok=True)
        cf_df.to_csv(crowdflower_path, index=False)
    else:
        cf_df = pd.read_csv(crowdflower_path)

    cf_df.drop(columns=['tweet_id', 'author'], inplace=True)

    # drop those with fewer instances
    cf_df = cf_df[~cf_df['sentiment'].isin(['anger', 'boredom'])]

    if sample:
        sampled = []
        for sentiment, group in cf_df.groupby('sentiment'):
            if len(group) <= 1500:
                sampled.append(group)
            else:
                sampled.append(group.sample(n=1500, random_state=42))

        cf_df = pd.concat(sampled)

    sentiment_counts = cf_df['sentiment'].value_counts()
    print('-----------------------------')
    print("# of data for each sentiment:")
    print(sentiment_counts)
    print('-----------------------------')

    if not os.path.exists(wassa2017_path):
        logging.warning('Wassa does not exist!!')
        exit(1)

    else:
        with open(wassa2017_path, "r", encoding="utf-8") as file:
            file.readline()
            wassa_X = []
            wassa_y = []
            for line in file:
                columns = line.strip().split("\t")
                wassa_X.append(columns[1])
                wassa_y.append('anger')

    wassa_df = pd.DataFrame({'sentiment': wassa_y,
                             'content': wassa_X})
    df = pd.concat([cf_df, wassa_df])
    return df


def split(df, test_size=0.2):
    train_data = {}
    test_data = {}
    for s in df['sentiment'].unique():
        data = df[df['sentiment'] == s]

        train_df, test_df = train_test_split(data, test_size=test_size, random_state=42)
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
