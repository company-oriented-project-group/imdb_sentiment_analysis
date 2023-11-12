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