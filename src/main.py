
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
