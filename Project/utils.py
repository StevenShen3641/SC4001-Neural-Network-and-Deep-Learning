import text_hammer as th
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def preprocess(X):
    return X.map(lambda x: x.lower().split("\n")).map(lambda x: [y.split(". ") for y in x]).map(
        lambda x: [i.replace('\'', '') for sl in x for i in sl if i != ''])


# Preprocess both train and test separately. Remove punctuations and other unwanted characters
def replace_punct(st, punct):
    for i in punct:
        if i == "..":
            st = st.replace("..", '.')
        elif i == "--" or i == '-':
            st = st.replace(i, ' ')
        else:
            st = st.replace(i, '')
    return st


def remove_special_content(df):
    # These are series of preprocessing
    df['content'] = df['content'].progress_apply(lambda x: th.cont_exp(x))  # you're -> you are; i'm -> i am
    df['content'] = df['content'].progress_apply(lambda x: th.remove_emails(x))
    df['content'] = df['content'].progress_apply(lambda x: th.remove_html_tags(x))
    df['content'] = df['content'].progress_apply(lambda x: th.remove_urls(x))
    df['content'] = df['content'].progress_apply(lambda x: th.remove_special_chars(x))
    df['content'] = df['content'].progress_apply(lambda x: th.remove_accented_chars(x))
    df['content'] = df['content'].progress_apply(lambda x: th.make_base(x))  # ran -> run,
    print("Number of words with stopwords:", df['content'].str.split().str.len().sum())
    df["content"] = df['content'].progress_apply(lambda x: remove_stopwords(x))
    print("Number of words without stopwords:", df['content'].str.split().str.len().sum())
