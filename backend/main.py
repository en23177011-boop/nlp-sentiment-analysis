from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from nltk.sentiment import SentimentIntensityAnalyzer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import nltk
import re
import sqlite3

# Download NLP data quietly
nltk.download('vader_lexicon', quiet=True)

# 1. RATE LIMITING: Set up the throttler to use the user's IP address
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sia = SentimentIntensityAnalyzer()

# 2. DATABASE: Initialize a lightweight SQLite database for Global Analytics
def init_db():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentiment TEXT,
            compound_score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 3. INPUT SANITIZATION: Secure the Pydantic model
class TextRequest(BaseModel):
    text: str

    @field_validator('text')
    def sanitize_input(cls, v):
        # Strip out HTML/XML tags to prevent Cross-Site Scripting (XSS) and injection vectors
        sanitized = re.sub(r'<[^>]*>', '', v)
        if not sanitized.strip():
            raise ValueError('Input cannot be empty or just malicious tags.')
        return sanitized

@app.post("/analyze")
@limiter.limit("5/minute") # Restrict to 5 requests per minute per IP
def analyze_sentiment(request: Request, text_req: TextRequest):
    scores = sia.polarity_scores(text_req.text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    # Log to Global Database
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries (sentiment, compound_score) VALUES (?, ?)", (sentiment, compound))
    conn.commit()
    conn.close()
        
    return {"sentiment": f"{sentiment} {'😃' if sentiment=='Positive' else '😞' if sentiment=='Negative' else '😐'}", "scores": scores}

# 4. GLOBAL ANALYTICS ENDPOINT
@app.get("/analytics")
def get_global_analytics():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), sentiment FROM queries GROUP BY sentiment")
    data = cursor.fetchall()
    conn.close()
    
    stats = {"Total Requests": sum(row[0] for row in data), "Breakdown": {row[1]: row[0] for row in data}}
    return stats