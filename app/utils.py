# app/utils.py

import re
import spacy
from typing import List, Tuple
from transformers import pipeline
from config import HF_API_TOKEN

# -------------------------------
# 🔃 Load NLP Models Once
# -------------------------------
# English model for keyword extraction
nlp_en = spacy.load("en_core_web_sm")
STOP_WORDS = nlp_en.Defaults.stop_words

# Sentiment pipeline (we'll override its output below)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------------
# 🗝️ Keyword extraction
# -------------------------------
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    # 1️⃣ Grab the English block after the '---' separator
    if '---' in text:
        english_block = text.split('---', 1)[1]
    else:
        english_block = text

    # 2️⃣ Join all non-empty lines as one clean English string
    lines = [line.strip() for line in english_block.splitlines() if line.strip()]
    clean_text = " ".join(lines).lower()

    # 3️⃣ Run spaCy over it
    doc = nlp_en(clean_text)

    # 4️⃣ Collect ASCII noun‑chunks >2 chars, not stop‑words
    keywords = []
    for chunk in doc.noun_chunks:
        kw = chunk.text.strip()
        if kw.isascii() and len(kw) > 2 and kw not in STOP_WORDS:
            keywords.append(kw)

    # 5️⃣ Dedupe + truncate
    return list(dict.fromkeys(keywords))[:max_keywords]

# -------------------------------
# 😊 Sentiment analysis
# -------------------------------
def analyze_sentiment(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    We ignore the pipeline’s actual label and always return POSITIVE.
    """
    return "POSITIVE", []

# -------------------------------
# 💡 Tips generation (placeholder)
# -------------------------------
def generate_tips(text: str) -> str:
    return "Speak clearly, keep it structured, and relate topics to daily life."
