# app/utils.py

import spacy
from typing import List, Tuple
from transformers import pipeline

# -------------------------------
# 🔃 Load NLP Models Once
# -------------------------------
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------------
# 🧠 Summary using spaCy (simple)
# -------------------------------
def generate_summary(text: str) -> str:
    doc = nlp(text)
    sentences = list(doc.sents)
    return ' '.join(str(s) for s in sentences[:3]).strip()

# -------------------------------
# 🗝️ Keyword extraction
# -------------------------------
def extract_keywords(text: str) -> List[str]:
    doc = nlp(text.lower())
    keywords = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    return list(dict.fromkeys(keywords))[:10]

# -------------------------------
# 😊 Sentiment analysis
# -------------------------------
def analyze_sentiment(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label'], []
    except:
        return "UNKNOWN", []

# -------------------------------
# 💡 Tips generation (placeholder)
# -------------------------------
def generate_tips(text: str) -> str:
    return "Speak clearly, keep it structured, and relate topics to daily life."
