import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import gensim
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from config import (
    TEXT_CLEANING_REGEX, TRAINING_SIZE, W2V_SG, W2V_HS, W2V_EPOCHS, W2V_WORKERS,
    W2V_VECTOR_SIZE, W2V_MIN_COUNT, W2V_WINDOW
)

class Word2Vec:
    def __init__(self, data_path):
        self.data_path = data_path
        self.movie_reviews_data = None
        self.train = None
        self.test = None
        self.stop_words = None
        self.snowball_stemmer = None
        self.w2v_model = None

    def load_data(self):
        self.movie_reviews_data = pd.read_csv(self.data_path, encoding='utf-8')

    def explore_data(self):
        print(self.movie_reviews_data.info())
        print(self.movie_reviews_data.shape)
        print(self.movie_reviews_data.head())

    def set_seaborn_style(self):
        sns.set(style="white", font_scale=1.2)
        plt.figure(figsize=(5, 4))
        sns.countplot(x='sentiment', data=self.movie_reviews_data)

    def preprocess_data(self):
        nltk.download('stopwords')
        self.stop_words = stopwords.words("english")
        self.stop_words.append('br')
        self.snowball_stemmer = SnowballStemmer("english")
        self.movie_reviews_data['review'] = self.movie_reviews_data['review'].apply(self.clean_and_tokenize_text)

    def clean_and_tokenize_text(self, text):
        tokens = []
        text = re.sub(TEXT_CLEANING_REGEX, ' ', str(text).lower()).strip()
        for token in text.split():
            if token not in self.stop_words:
                tokens.append(self.snowball_stemmer.stem(token))
        return " ".join(tokens)

    def split_data(self):
        self.train, self.test = train_test_split(self.movie_reviews_data, test_size=1 - TRAINING_SIZE, random_state=42)

    def visualize_word_clouds(self):
        positive_sentiments = self.train[self.train['sentiment'] == POSITIVE_SENTIMENT]
        negative_sentiments = self.train[self.train['sentiment'] == NEGATIVE_SENTIMENT]

        plt.figure(figsize=(12, 12))
        word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(positive_sentiments['review']))
        plt.title('Word Cloud for Positive Reviews')
        plt.imshow(word_cloud, interpolation='bilinear')

        plt.figure(figsize=(12, 12))
        word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(negative_sentiments['review']))
        plt.title('Word Cloud for Negative Reviews')
        plt.imshow(word_cloud, interpolation='bilinear')

    def build_word2vec_model(self):
        documents = [text.split() for text in self.train['review']]
        self.w2v_model = gensim.models.Word2Vec(sg=W2V_SG, hs=W2V_HS, workers=W2V_WORKERS,
                                                vector_size=W2V_VECTOR_SIZE, min_count=W2V_MIN_COUNT, window=W2V_WINDOW)
        self.w2v_model.build_vocab(documents)
        words = self.w2v_model.wv.index_to_key
        vocabulary_size = len(words)
        print("Vocabulary size:", vocabulary_size)
        self.w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCHS)

    def save_word2vec_model(self, model_path):
        self.w2v_model.save(model_path)

    def test_word2vec_model(self, model_path):
        test_model = gensim.models.Word2Vec.load(model_path)
        print(test_model.wv.most_similar("love"))

if __name__ == "__main__":
    nlp_project = Word2Vec(data_path='./IMDB-example-dataset.csv')
    nlp_project.load_data()
    nlp_project.explore_data()
    nlp_project.set_seaborn_style()
    nlp_project.preprocess_data()
    nlp_project.split_data()
    nlp_project.visualize_word_clouds()
    nlp_project.build_word2vec_model()
    nlp_project.save_word2vec_model("word2vec.model")
    nlp_project.test_word2vec_model("word2vec.model")
