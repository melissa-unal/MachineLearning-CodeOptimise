import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer
from spacy.lang.it import Italian
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import time
stop_time=time.time()


TRAIN_FILE = ''
TEST_FILE = ''
TXT_COL = 'text'
LBL_COL = 'label'

parser=Italian()

def tokenize(text):

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_'''
    for char in text:
        if char in punctuations:
            text = text.replace(char, "")

    '''numbers = '0123456789'
    for num in text:
        if num in numbers:
            text = text.replace(num, "")'''

    tokens = parser(text.lower())
    tokens = [word.lemma_.strip() for word in tokens]

    return tokens
    #return nltk.WordPunctTokenizer().tokenize(text.lower())
    # return nltk.TweetTokenizer().tokenize(text.lower())
    # return word_tokenize(text.lower())

def get_representation(vocabulary, how_many):

    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}

    for i, iterator in enumerate(most_comm):
        cuv = iterator[0]
        wd2idx[cuv] = i
        idx2wd[i] = cuv

    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):

    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):

    features = np.zeros(len(wd2idx))
    tokenz = tokenize(text)
    for tok in tokenz:
        if tok in wd2idx:
            features[wd2idx[tok]] += 1

    return features



def corpus_to_bow(corpus, wd2idx):

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text,wd2idx))

    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):

    test_df = pd.read_csv('test.csv')
    f = open("submis.csv", "w")

    f.write("id,label")
    for i in range(len(predictions)):
        f.write("\n%d,%d" % (test_df['id'][0], predictions[i]))
        test_df['id'] += 1

    pass



train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

toate_cuvintele = get_corpus_vocabulary(corpus)

wd2idx,idx2wd = get_representation(toate_cuvintele, 10000)
labels = train_df['label'].values

data = corpus_to_bow(corpus,wd2idx)
test_data = corpus_to_bow(test_df['text'], wd2idx)


def cross_validate(k, data, labels):
    segment_size = int(len(labels) / k)
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), segment_size):
        indici_valid = indici[i:i + segment_size]
        left_side = indici[:i]
        right_side = indici[i + segment_size:]
        indici_train = np.concatenate([left_side, right_side])
        train = data[indici_train]
        valid = data[indici_valid]
        y_train = labels[indici_train]
        y_valid = labels[indici_valid]
        yield train, valid, y_train, y_valid

'''mnb = MultinomialNB()
mnb.fit(data,labels)
predictii = mnb.predict(test_data)
write_prediction('submis.csv', predictii)'''

mnb = MultinomialNB()
n = np.zeros((2,2))

for x_train, x_valid, y_train, y_valid in cross_validate(10, data, labels):
    mnb.fit(x_train, y_train)
    y_pred = mnb.predict(x_valid)
    m = confusion_matrix(y_pred, y_valid, labels=[0,1])
    n = n + m
    print("\n")
    print(n)
    acc = metrics.accuracy_score(y_valid, y_pred)
    print(acc)

predictii = mnb.predict(test_data)
write_prediction('submis.csv', predictii)


#print("Timp de executie: %.2f secunde"%(time.time()-stop_time))