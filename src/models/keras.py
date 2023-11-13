import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re, os
import gensim
import itertools
import time, sys
import tensorflow as tf

sys.path.append('../')
# from config import (
#     TEXT_CLEANING_REGEX, TRAINING_SIZE, SEQUENCE_LENGTH, W2V_SG, W2V_HS, W2V_EPOCHS, W2V_WORKERS,
#     W2V_VECTOR_SIZE, W2V_MIN_COUNT, W2V_WINDOW, POSITIVE_SENTIMENT, NEUTRAL_SENTIMENT, NEGATIVE_SENTIMENT
# )

# Global Constants
TEXT_CLEANING_REGEX = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
TRAINING_SIZE = 0.8

# Word2Vec Global Parameters
W2V_SG = 1
W2V_HS = 0
W2V_EPOCHS = 32
W2V_WORKERS = 8
W2V_VECTOR_SIZE = 300
W2V_MIN_COUNT = 10
W2V_WINDOW = 4

# Keras Configuration
EPOCHS = 16
SEQUENCE_LENGTH = 300
BATCH_SIZE = 1024

# Sentiment Labels
SENTIMENT_LEVELS = (0.3, 0.7)
NEGATIVE_SENTIMENT = "negative"
NEUTRAL_SENTIMENT = "neutral"
POSITIVE_SENTIMENT = "positive"

# Export Models
TOKENIZER_MODEL = "tokenizer.pkl"
KERAS_MODEL = "model.h5"
ENCODER_MODEL = "encoder.pkl"
WORD2VEC_MODEL = "model.w2v"

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def build_the_model():
    data = pd.read_csv('IMDB-example-dataset.csv')
    print(data.info())
    print(data.shape)
    print(data.head())

    sns.set(style = "white" , font_scale = 1.2)
    plt.figure(figsize=(5,4))
    sns.countplot(x='sentiment',data = data)

    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words("english")
    stop_words.append('br')
    stemmer = SnowballStemmer("english")

    def preprocess(text, stem=False):
        """ Remove links and special characters """
        text = re.sub(TEXT_CLEANING_REGEX, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    data.review = data.review.apply(lambda x: preprocess(x))
    print(data.head())

    # Split into training set and test set 4:1
    train, test = train_test_split(data, test_size = 1 - TRAIN_SIZE, random_state = 84)
    print("train data size:", len(train))
    print("test data size:", len(test))

    print(train['sentiment'].value_counts())
    print(test['sentiment'].value_counts())

    Positive_sent = train[train['sentiment']=='positive']
    Negative_sent = train[train['sentiment']=='negative']

    plt.figure(figsize = (10,10)) #Positive Review Text
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(Positive_sent.review))
    plt.title('Positive Review Text Wordcloud')
    plt.imshow(wc , interpolation = 'bilinear')

    plt.figure(figsize = (10,10)) # Negative Review Text
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800).generate(" ".join(Negative_sent.review))
    plt.title('Negative Review Text Wordcloud')
    plt.imshow(wc , interpolation = 'bilinear')

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train.review)#实现分词，形成词汇表

    vocab_size = len(tokenizer.word_index) + 1 #词汇表大小，因为加上为未知词，维度加1
    print("Total words", vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(train.review), maxlen = SEQUENCE_LENGTH)#输出向量序列并进行补全
    x_test = pad_sequences(tokenizer.texts_to_sequences(test.review), maxlen = SEQUENCE_LENGTH)

    print(x_train)

    encoder = LabelEncoder()
    encoder.fit(train.sentiment.tolist()) #positive和negative加标签

    y_train = encoder.transform(train.sentiment.tolist())
    y_test = encoder.transform(test.sentiment.tolist())

    y_train = y_train.reshape(-1,1) #转置
    y_test = y_test.reshape(-1,1)

    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print()
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    # Generate embedding matrix
    # Each column of this matrix represents a different vector represented by each different word in the vocabulary
    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    print(embedding_matrix.shape)

    # Embedding layer, a network layer in the first layer of the model
    embedding_layer = Embedding(vocab_size,
                        W2V_SIZE,
                        weights=[embedding_matrix],
                        input_length = SEQUENCE_LENGTH,
                        trainable=False)

    # Sequential Model structure: Linear stack of layers.
    # It is a simple linear structure with no redundant branches and is a stack of multiple network layers.
    model = tf.keras.models.Sequential()
    model.add(embedding_layer) #Add embedding layer
    model.add(Dropout(0.5)) #The output of the first layer will implement Dropout regularization for the second layer
    model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))#recurrent_dropout controls the neuron disconnection ratio of the linear transformation of the recurrent state
    model.add(Dense(1, activation='sigmoid'))


    print(model.summary())

    print(len(model.layers))

    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    callbacks = [ ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=0, min_lr=0),
                  EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

    history = model.fit(x_train, y_train,
                        batch_size = BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_split = 0.1,
                        verbose = 1,
                        callbacks = callbacks)

    tf_model_path = r'./model.keras'
    model.save(tf_model_path)

    # model = tf.keras.models.load_model(tf_model_path)  # Uncomment this to load
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print()
    print("ACCURACY:",score[1])
    print("LOSS:",score[0])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Model Evaluation
def decode_sentiment(score, include_neutral = True):
    if include_neutral:
        label = NEUTRAL_SENTIMENT
        if score <= SENTIMENT_LEVELS[0]:
            label = NEGATIVE_SENTIMENT
        elif score >= SENTIMENT_LEVELS[1]:
            label = POSITIVE_SENTIMENT

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral = True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}


def predict_preparation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'keras.model')
    model = tf.keras.models.load_model(model_path)
    # model = tf.keras.models.load_model('keras.model')
    data = pd.read_csv('IMDB-example-dataset-processed.csv')
    train, test = train_test_split(data, test_size = 1 - TRAINING_SIZE, random_state = 84)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train.review)
    return model, tokenizer

def predict_sentiment(text, include_neutral = True):
    start_at = time.time()
    # Tokenize text
    model, tokenizer = predict_preparation()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}

