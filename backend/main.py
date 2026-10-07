from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from nltk.sentiment import SentimentIntensityAnalyzer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from better_profanity import profanity
from spacy import displacy # NEW: Import displacy for the SVG graph
import nltk
import re
import sqlite3
import spacy

nltk.download('vader_lexicon', quiet=True)
nlp = spacy.load("en_core_web_sm")
profanity.load_censor_words()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sia = SentimentIntensityAnalyzer()

def init_db():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY AUTOINCREMENT, sentiment TEXT, compound_score REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, sentiment TEXT)''')
    conn.commit()
    conn.close()

init_db()

class TextRequest(BaseModel):
    text: str
    @field_validator('text')
    def sanitize(cls, v):
        sanitized = re.sub(r'<[^>]*>', '', v)
        if not sanitized.strip(): raise ValueError('Invalid input.')
        return sanitized

@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_sentiment(request: Request, text_req: TextRequest):
    has_bad_words = profanity.contains_profanity(text_req.text)

    scores = sia.polarity_scores(text_req.text)
    compound = scores['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        
    doc = nlp(text_req.text)
    extracted_keywords = [chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.root.pos_ != "PRON" and len(chunk.text) > 2]
    
    linguistics = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            linguistics.append({"word": token.text, "pos": token.pos_, "lemma": token.lemma_, "morph": str(token.morph)})
            
    # NEW 1: Sentence-by-Sentence Flow Analysis
    sentence_flow = []
    for i, sent in enumerate(doc.sents):
        sent_score = sia.polarity_scores(sent.text)['compound']
        sentence_flow.append({"sentence_num": i + 1, "text": sent.text, "score": sent_score})
        
    # NEW 2: Generate the Dependency Tree SVG
    # We use custom options so it matches your dark corporate UI theme
    dep_svg = displacy.render(doc, style="dep", page=False, options={
        "compact": True, 
        "color": "#cbd5e1", 
        "bg": "transparent",
        "font": "Segoe UI"
    })
        
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO queries (sentiment, compound_score) VALUES (?, ?)", (sentiment, compound))
    for keyword in extracted_keywords:
        cursor.execute("INSERT INTO entities (name, sentiment) VALUES (?, ?)", (keyword, sentiment))
    conn.commit()
    conn.close()
        
    return {
        "sentiment": sentiment, 
        "scores": scores, 
        "extracted_topics": extracted_keywords,
        "linguistics": linguistics,
        "has_profanity": has_bad_words,
        "sentence_flow": sentence_flow, # Send flow coordinates
        "dependency_svg": dep_svg       # Send raw SVG image
    }

@app.get("/analytics")
def get_global_analytics():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), sentiment FROM queries GROUP BY sentiment")
    sentiment_data = cursor.fetchall()
    cursor.execute("SELECT name, COUNT(*) as count FROM entities GROUP BY name ORDER BY count DESC LIMIT 30")
    word_cloud = [{"text": r[0], "value": r[1]} for r in cursor.fetchall()]
    cursor.execute('''SELECT name, SUM(CASE WHEN sentiment='Positive' THEN 1 ELSE 0 END) as pos, SUM(CASE WHEN sentiment='Negative' THEN 1 ELSE 0 END) as neg, COUNT(*) as total FROM entities GROUP BY name ORDER BY total DESC LIMIT 5''')
    correlations = [{"topic": r[0], "pos": r[1], "neg": r[2], "total": r[3]} for r in cursor.fetchall()]
    conn.close()
    return {"Total Requests": sum(row[0] for row in sentiment_data), "Breakdown": {row[1]: row[0] for row in sentiment_data}, "WordCloud": word_cloud, "Correlations": correlations}