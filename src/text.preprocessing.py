import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('data/movie_reviews.csv', encoding='utf-8')
print(df.head())

df['review_text'] = df['review_text'].apply(lambda text: word_tokenize(text))

stemmer = PorterStemmer()
df['review_text'] = df['review_text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

stop_words = set(stopwords.words('english'))
df['review_text'] = df['review_text'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

df.to_csv('data/preprocessed_review_texts.csv', index=False)