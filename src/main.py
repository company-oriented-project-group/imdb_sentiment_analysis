# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse
# import pandas as pd
# from textblob import TextBlob

# app = FastAPI()

# # Load CSV file
# df = pd.read_csv("data.csv")

# # Perform sentiment analysis
# df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)


# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     return {"message": "Welcome to Sentiment Analysis using FastAPI!"}


# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     df_new = pd.read_csv(file.file)
#     df_new['sentiment'] = df_new['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
#     return df_new.to_dict()


# @app.get("/sentiment/{text}")
# async def read_item(text: str):
#     sentiment_score = TextBlob(text).sentiment.polarity
#     sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
#     return {"text": text, "sentiment": sentiment_label, "sentiment_score": sentiment_score}

from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

app = FastAPI()

# Store posted data in-memory
posted_data = []

nltk.download('vader_lexicon')

class SentimentRequest(BaseModel):
    review_text: str


@app.post("/sentiment/")
def analyze_sentiment(request_data: SentimentRequest):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(request_data.review_text)
    # Store the posted data and sentiment scores
    posted_data.append({"review_text": request_data.review_text, "sentiment_scores": sentiment_scores})
    return sentiment_scores


@app.get("/sentiment/")  # Add a GET method to the /sentiment/ route
def get_sentiment_data():
    return posted_data

if __name__ == "__main__":
    import uvicorn

    # Read data from CSV file
    data = pd.read_csv("data.csv")

    # Post each review_text to the /sentiment route
    for review_text in data["review_text"]:
        payload = SentimentRequest(review_text=review_text)
        analyze_sentiment(payload)

    # Run the FastAPI application
    uvicorn.run(app, host="localhost", port=8000)
