import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

df = pd.read_csv('./data/preprocessed_review_texts.csv')

print(df.shape)
df.head()

#the output is (500, 1) meaning there are 500 rows and 1 argument, the review

example = df['review_text'][6]
print(example)
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()