from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from nltk.sentiment import SentimentIntensityAnalyzer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from better_profanity import profanity
from spacy import displacy
import nltk
import re
import sqlite3
import spacy

# Download NLP data
nltk.download('vader_lexicon', quiet=True)
nlp = spacy.load("en_core_web_sm")

# Initialize profanity filter
profanity.load_censor_words()

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

EMOTION_LEXICON = {
    "Joy": ["thrilled", "fantastic", "brilliant", "happy", "excellent", "flawless", "seamless", "beautiful"],
    "Frustration": ["frustrating", "broken", "laggy", "shit", "terrible", "crash", "crashed", "fails"],
    "Urgency": ["rapid", "immediate", "urgent", "quickly", "hotfix", "critical", "soon"],
    "Disappointment": ["disappointment", "sad", "failed", "unfortunately", "bad"]
}

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

# NEW: Data model for bulk CSV processing
class BatchRequest(BaseModel):
    texts: list[str]

@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_sentiment(request: Request, text_req: TextRequest):
    has_bad_words = profanity.contains_profanity(text_req.text)
    censored_text = profanity.censor(text_req.text, '*')

    scores = sia.polarity_scores(text_req.text)
    compound = scores['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        
    doc = nlp(text_req.text)
    extracted_keywords = [chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.root.pos_ != "PRON" and len(chunk.text) > 2]
    
    absa_pairs = []
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ in ["NOUN", "PROPN"]:
            absa_pairs.append({"aspect": token.head.text, "descriptor": token.text})

    emotions_detected = set()
    text_lower = text_req.text.lower()
    for emotion, keywords in EMOTION_LEXICON.items():
        if any(word in text_lower for word in keywords):
            emotions_detected.add(emotion)
            
    linguistics = [{"word": t.text, "pos": t.pos_, "lemma": t.lemma_, "morph": str(t.morph)} for t in doc if not t.is_punct and not t.is_space]
    sentence_flow = [{"sentence_num": i + 1, "text": sent.text, "score": sia.polarity_scores(sent.text)['compound']} for i, sent in enumerate(doc.sents)]
    
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
        "original_text": text_req.text, "censored_text": censored_text, "sentiment": sentiment, 
        "scores": scores, "extracted_topics": extracted_keywords, "absa": absa_pairs,
        "emotions": list(emotions_detected), "linguistics": linguistics, "has_profanity": has_bad_words,
        "sentence_flow": sentence_flow, "dependency_svg": dep_svg
    }

# NEW: Bulk processing endpoint for CSV uploads
@app.post("/analyze-bulk")
@limiter.limit("2/minute")
def analyze_bulk_sentiment(request: Request, batch_req: BatchRequest):
    # Cap at 100 to prevent Render memory timeouts
    texts_to_process = batch_req.texts[:100] 
    results = {"Positive": 0, "Neutral": 0, "Negative": 0, "Total": 0}
    
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    
    for text in texts_to_process:
        if not text or not str(text).strip(): continue
        score = sia.polarity_scores(str(text))['compound']
        sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        
        results[sentiment] += 1
        results["Total"] += 1
        cursor.execute("INSERT INTO queries (sentiment, compound_score) VALUES (?, ?)", (sentiment, score))
        
    conn.commit()
    conn.close()
    
    return results

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