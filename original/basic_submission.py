import nltk
import pandas as pd
import random
import numpy as np
from collections import Counter


def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    return nltk.WordPunctTokenizer().tokenize(text)


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wd2idx     @  che  .   ,   di  e
    idx2wd     0   1   2   3   4   5
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
           0   1   2   3   4   5
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)
    return all_features


def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed


def split(data, labels, procentaj_valid=0.25):
    '''Split data and labels into train and valid by procentaj_valid.
    75% train, 25% valid
    Important! shuffle the data before splitting.
    '''
    return train, valid, y_train, y_valid


def cross_validate(k, data, labels):
    '''Split the data into k chunks.
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] for train
    '''
    yield train, valid, y_train, y_valid


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label'].values

test_data = corpus_to_bow(test_df['text'], wd2idx)
print(test_data.shape)

predictii = np.ones(len(test_data))
write_prediction('sample_submission.csv', predictii)


