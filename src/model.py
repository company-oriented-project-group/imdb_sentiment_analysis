import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

plt.style.use('ggplot')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

df = pd.read_csv('./data/preprocessed_review_texts.csv')

print(df.shape)
df.head()

# the output is (500, 1) meaning there are 500 rows and 1 argument, the review

example = df['review_text'][6]
print(example)
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

sia = SentimentIntensityAnalyzer()

# example sentance
sia.polarity_scores('I am so happy!')


nltk.download('vader_lexicon')

class SentimentRequest(BaseModel):
    sentence: str
