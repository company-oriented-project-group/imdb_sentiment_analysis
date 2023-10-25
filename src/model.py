import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import gensim
import time
import tensorflow as tf
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Global Constants
TEXT_CLEANING_REGEX = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
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

# Load the dataset
movie_reviews_data  = pd.read_csv('./IMDB-example-dataset.csv', encoding='utf-8')
print(movie_reviews_data.info())
print(movie_reviews_data.shape)
print(movie_reviews_data.head())

# Set seaborn style and font scale
sns.set(style="white", font_scale=1.2)
plt.figure(figsize=(5, 4))
sns.countplot(x='sentiment', data=movie_reviews_data )

# Data Preprocessing and Exploratory Analysis
nltk.download('stopwords')
stop_words = stopwords.words("english")
stop_words.append('br')
snowball_stemmer = SnowballStemmer("english")

def clean_and_tokenize_text(text, stem=False):
    tokens = []
    text = re.sub(TEXT_CLEANING_REGEX, ' ', str(text).lower()).strip()
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(snowball_stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

movie_reviews_data['review'] = movie_reviews_data ['review'].apply(lambda text: clean_and_tokenize_text(text))

movie_reviews_data.head()

# Split the dataset into training and testing sets (80% training, 20% testing)
train, test = train_test_split(movie_reviews_data , test_size=1 - TRAINING_SIZE, random_state=42)
print("Train data size:", len(train))
print("Test data size:", len(test))

print(train['sentiment'].value_counts())
print(test['sentiment'].value_counts())

positive_sentiments = train[train['sentiment'] == POSITIVE_SENTIMENT]
negative_sentiments = train[train['sentiment'] == NEGATIVE_SENTIMENT]

# Visualize Word Cloud for Positive Reviews
plt.figure(figsize=(12, 12))
word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(positive_sentiments['review']))
plt.title('Word Cloud for Positive Reviews')
plt.imshow(word_cloud, interpolation='bilinear')

# Visualize Word Cloud for Negative Reviews
plt.figure(figsize=(12, 12))
word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(negative_sentiments['review']))
plt.title('Word Cloud for Negative Reviews')
plt.imshow(word_cloud, interpolation='bilinear')

# Building the Word2Vec model
documents = [text.split() for text in train['review']]
w2v_model = gensim.models.Word2Vec(sg=W2V_SG,
                                   hs=W2V_HS,
                                   workers=W2V_WORKERS,
                                   vector_size=W2V_VECTOR_SIZE,
                                   min_count=W2V_MIN_COUNT,
                                   window=W2V_WINDOW,)
w2v_model.build_vocab(documents)
words = w2v_model.wv.index_to_key
vocabulary_size = len(words)
print("Vocabulary size:", vocabulary_size)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCHS)

print(w2v_model.wv.most_similar("love"))

# Save the Word2Vec model for future use
w2v_model.save("word2vec.model")

# Verify the functionality of the saved Word2Vec model
test_model = gensim.models.Word2Vec.load(word2vec.model)
print(test_model.wv.most_similar("love"))
