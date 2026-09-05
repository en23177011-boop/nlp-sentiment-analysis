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
import spacy

# Download NLP data quietly
nltk.download('vader_lexicon', quiet=True)

# Load spaCy English model for Context Extraction
nlp = spacy.load("en_core_web_sm")

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

def init_db():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    # Table for overall sentiment
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentiment TEXT,
            compound_score REAL
        )
    ''')
    # NEW: Table for extracted context (Entities/Nouns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class TextRequest(BaseModel):
    text: str

    @field_validator('text')
    def sanitize_input(cls, v):
        sanitized = re.sub(r'<[^>]*>', '', v)
        if not sanitized.strip():
            raise ValueError('Input cannot be empty or just malicious tags.')
        return sanitized

@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_sentiment(request: Request, text_req: TextRequest):
    # 1. Sentiment Analysis
    scores = sia.polarity_scores(text_req.text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    # 2. Context Extraction (NER)
    doc = nlp(text_req.text)
    # Extract noun chunks, ignore plain pronouns, and make lowercase
    extracted_keywords = [
        chunk.text.lower().strip() 
        for chunk in doc.noun_chunks 
        if chunk.root.pos_ != "PRON" and len(chunk.text) > 2
    ]
        
    # 3. Log to Global Database
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries (sentiment, compound_score) VALUES (?, ?)", (sentiment, compound))
    
    for keyword in extracted_keywords:
        cursor.execute("INSERT INTO entities (name) VALUES (?)", (keyword,))
        
    conn.commit()
    conn.close()
        
    return {
        "sentiment": f"{sentiment} {'😃' if sentiment=='Positive' else '😞' if sentiment=='Negative' else '😐'}", 
        "scores": scores,
        "extracted_topics": extracted_keywords
    }

@app.get("/analytics")
def get_global_analytics():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    
    # Get sentiment breakdown
    cursor.execute("SELECT COUNT(*), sentiment FROM queries GROUP BY sentiment")
    sentiment_data = cursor.fetchall()
    
    # NEW: Get Top 5 most mentioned entities
    cursor.execute("SELECT name, COUNT(*) as count FROM entities GROUP BY name ORDER BY count DESC LIMIT 5")
    top_entities = cursor.fetchall()
    
    conn.close()
    
    stats = {
        "Total Requests": sum(row[0] for row in sentiment_data), 
        "Breakdown": {row[1]: row[0] for row in sentiment_data},
        "Top Topics": [{"topic": row[0], "count": row[1]} for row in top_entities]
    }
    return stats