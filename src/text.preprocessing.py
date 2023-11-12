import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Read the movie reviews data
df = pd.read_csv('imdb_data/movie_reviews.csv', encoding='utf-8')
print(df.head())
# Tokenize the review text
df['review_text'] = df['review_text'].apply(lambda text: word_tokenize(text))
# Remove stopwords
stop_words = set(stopwords.words('english'))
df['review_text'] = df['review_text'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
# Save the preprocessed data
df.to_csv('imdb_data/preprocessed_review_texts.csv', index=False)