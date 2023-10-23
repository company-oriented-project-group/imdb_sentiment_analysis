import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer  # Import the Snowball Stemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Read the movie reviews data
df = pd.read_csv('movie_reviews.csv', encoding='utf-8')
print(df.head())

# Tokenize the review text
df['review_text'] = df['review_text'].apply(lambda text: word_tokenize(text))

# Initialize the Snowball Stemmer
stemmer = SnowballStemmer('english')

# Apply stemming to each token in the 'review_text'
df['review_text'] = df['review_text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['review_text'] = df['review_text'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

# Save the preprocessed data
df.to_csv('preprocessed_review_texts.csv', index=False)