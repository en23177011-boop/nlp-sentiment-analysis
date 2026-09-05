from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os

# Download the required NLP dataset quietly
nltk.download('vader_lexicon', quiet=True)

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sia = SentimentIntensityAnalyzer()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    scores = sia.polarity_scores(request.text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive 😃"
    elif compound <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"
        
    return {"sentiment": sentiment, "scores": scores}
