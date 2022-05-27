import os
import re
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
from keras.models import load_model
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
from keras.layers import LSTM
from keras.layers.advanced_activations import Softmax
import gensim.models.keyedvectors as word2vec  # need to use due to depreceated model
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from nltk.corpus import stopwords  # to get collection of stopwords
from sklearn.model_selection import train_test_split  # for splitting dataset
import tensorflow as tf
import re
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

pd.set_option('display.max_colwidth', None)

ar_stopwords = stopwords.words('arabic')


# print("Stop words: ", ar_stopwords)
# print("length of stopwords is: ", len(ar_stopwords))


def process_text(text):
    stemmer = nltk.ISRIStemmer()
    word_list = nltk.word_tokenize(text)
    # remove arabic stopwords
    word_list = [w for w in word_list if not w in ar_stopwords]
    # remove digits
    word_list = [w for w in word_list if not w.isdigit()]
    # stemming
    word_list = [stemmer.stem(w) for w in word_list]
    return ' '.join(word_list)


def clean_text(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى",
              "\\", '\n', '\t', '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", ""
        , "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']
    # remove tashkeel
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, "", text)
    longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(longation, subst, text)
    text = re.sub(r"[^\w\s]", '', text)
    # remove english words
    text = re.sub(r"[a-zA-Z]", '', text)
    # remove spaces
    text = re.sub(r"\d+", ' ', text)
    text = re.sub(r"\n+", ' ', text)
    text = re.sub(r"\t+", ' ', text)
    text = re.sub(r"\r+", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    # remove repetetions
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()
    return process_text(text)


def execute(tweet):
    # Read the dataset and print it
    data = pd.read_csv('ArSAS.csv')
    # print(data.shape)
    # print(data.dtypes)
    # data.head()

    data.drop(columns=['#Tweet_ID', 'Topic', 'Sentiment_label_confidence', 'Speech_act_label',
                       'Speech_act_label_confidence'], inplace=True)
    # print(data.shape)
    # data.head()

    tweets = data['Tweet_text']
    labels = data['Sentiment_label']

    labels[data.Sentiment_label == 'Positive'] = 1
    labels[data.Sentiment_label == 'Negative'] = -1
    labels[data.Sentiment_label == 'Neutral'] = 2

    labels_count = labels.value_counts()
    labels_count.plot(kind="bar")
    # print(labels.value_counts())

    # Only include postive, negative and neutral polarity:
    data = data[data.Sentiment_label.isin([-1, 1, 2])]

    data['cleaned_text'] = tweets.apply(clean_text)
    data = data[data.cleaned_text != ""]
    # data.head(10)

    # Balance input data (same size postive, negative, and neutral tweets):

    min_sample = data.groupby(['Sentiment_label']).count().Tweet_text.min()
    input_data = pd.concat([data[data.Sentiment_label == 1].head(min_sample),
                            data[data.Sentiment_label == -1].head(min_sample),
                            data[data.Sentiment_label == 2].head(min_sample)
                            ])

    X = input_data.cleaned_text.values  # training paramter
    Y = np.asarray(input_data.Sentiment_label.values).astype('float32')  # prediction paramter
    Y = Y.clip(0, 2)

    num_words = 10000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X)

    X = tokenizer.texts_to_sequences(X)

    maxlen = 300
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    total_words = len(tokenizer.word_index) + 1

    seed = 42
    X_train, X_test, label_train, label_test = train_test_split(X, Y, test_size=0.1, random_state=seed)

    NUM_WORDS = 10000
    EMBEDDING_DIM = 300
    vocabulary_size = 10000

    model = Sequential()

    # embedding layer
    model.add(Embedding(total_words, EMBEDDING_DIM, input_length=maxlen))

    # convLayer1
    model.add(layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # convLayer2
    model.add(layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))

    # convLayer3
    model.add(layers.Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=8))

    # lstm
    model.add(layers.LSTM(64, dropout=0.2))
    model.add(layers.Dense((128), activation='relu'))
    model.add(layers.Dense((64), activation='relu'))
    model.add(layers.Dense((3), activation='softmax'))

    # dense
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.summary()

    # history = model.fit(X_train, label_train,
    #                    epochs=30,
    #                    verbose=True,
    #                    validation_data=(X_test, label_test),
    #                    batch_size=64)

    # loss, accuracy = model.evaluate(X_train, label_train, verbose=True)
    # print("Training Accuracy: {:.4f}".format(accuracy))

    # loss_val, accuracy_val = model.evaluate(X_test, label_test, verbose=True)
    # print("Testing Accuracy:  {:.4f}".format(accuracy_val))

    # pickle.dump(model, open('model.pkl', 'wb'))
    # pickled_model = pickle.load(open('model.pkl', 'rb'))
    model = load_model('cn_lstm')
    # model = pickle.load(open('model.pkl', 'rb'))

    sentiment = ['Negative', 'Positive', 'Neutral']
    # tweet = str(input('Enter Tweet: '))
    tweets = clean_text(tweet)

    words = tweets.split(' ')
    filtered = [w for w in words if w not in ar_stopwords]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]

    # print('Filtered: ', filtered)
    tokenize_words = tokenizer.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=300, padding='post', truncating='post')
    # print(tokenize_words)
    result = model.predict(tokenize_words)
    # print(result)
    # print(sentiment[np.around(result, decimals=0).argmax(axis=1)[0]])
    return sentiment[np.around(result, decimals=0).argmax(axis=1)[0]]
