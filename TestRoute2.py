# import unicodedata
# from flask import Flask, request, jsonify,url_for
# import urllib.parse
# from flask_cors import CORS
# from werkzeug.utils import secure_filename
# from flask_mail import Mail, Message
# import json
# from sqlalchemy import desc
# from sqlalchemy import func
# from config import SECRET_KEY
# import re
# import requests
# from langdetect import detect
# from deep_translator import GoogleTranslator
# from sqlalchemy.exc import IntegrityError
# from config import Session
# from models import User, Khutbah,Intention,PendingUser
# from config import MAIL_SERVER, MAIL_PORT, MAIL_USE_TLS, MAIL_USERNAME, MAIL_PASSWORD, MAIL_DEFAULT_SENDER
# from app.utils import generate_summary, extract_keywords, analyze_sentiment, generate_tips
# import os
# import uuid
# import secrets
# from datetime import datetime
# from authlib.integrations.flask_client import OAuth
# from config import GOOGLE_CLIENT_ID
# from config import GOOGLE_CLIENT_SECRET
# from config import GOOGLE_DISCOVERY_URL
# from config import HADITH_API_KEY
# from werkzeug.security import generate_password_hash
# from werkzeug.security import check_password_hash
#
#
# from flask import Flask, request, jsonify
# from pydub import AudioSegment, silence
# import whisper, torch, os, uuid, re, json
# from glob import glob
# from transformers import MarianTokenizer, MarianMTModel
#
#
# # Whisper AI
# import whisper
# import torch
#
# import stripe
# from config import STRIPE_SECRET_KEY
# stripe.api_key = STRIPE_SECRET_KEY
#
#
#
# # ---------------------------
# # 🤖 Whisper Model Setup
# # ---------------------------
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ASR_MODEL = whisper.load_model("medium", device=device)
# print("Whisper model loaded")
#
# # Choose your MT model here; Qur’an‑specific works for surahs,
# # but you can swap in a generic "Helsinki-NLP/opus-mt-ar-en" for khutba.
# MT_MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
# MT_TOKENIZER       = MarianTokenizer.from_pretrained(MT_MODEL_NAME)
# MT_MODEL           = MarianMTModel.from_pretrained(MT_MODEL_NAME)
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
#
# # ---------------------------
# # 📦 App Initialization
# # ---------------------------
# app = Flask(__name__)
# app.secret_key = SECRET_KEY
# CORS(app)
#
# # ---------------------------
# # 📧 Mail Configuration
# # ---------------------------
# app.config.update(
#     MAIL_SERVER=MAIL_SERVER,
#     MAIL_PORT=MAIL_PORT,
#     MAIL_USE_TLS=MAIL_USE_TLS,
#     MAIL_USERNAME=MAIL_USERNAME,
#     MAIL_PASSWORD=MAIL_PASSWORD,
#     MAIL_DEFAULT_SENDER=MAIL_DEFAULT_SENDER,
# )
# mail = Mail(app)
#
# # ---------------------------
# # 🤖 Google Authetication
# # ---------------------------
# # Initialize OAuth
# oauth = OAuth(app)
# google = oauth.register(
#     name='google',
#     client_id=GOOGLE_CLIENT_ID,
#     client_secret=GOOGLE_CLIENT_SECRET,
#     server_metadata_url=GOOGLE_DISCOVERY_URL,
#     client_kwargs={'scope': 'openid email profile'}
# )
#
#
# def chunk_audio(path, by_silence=False, min_silence_len=700, silence_thresh=-40):
#     try:
#         audio = AudioSegment.from_file(path)
#         os.makedirs(f"{UPLOAD_FOLDER}/chunks", exist_ok=True)
#         segs = (silence.split_on_silence(audio, min_silence_len=min_silence_len,
#                                           silence_thresh=silence_thresh)
#                 if by_silence else
#                 [audio[i:i+60_000] for i in range(0, len(audio), 60_000)])
#         paths = []
#         for i, seg in enumerate(segs):
#             out = f"{UPLOAD_FOLDER}/chunks/chunk_{i:03d}.wav"
#             seg.export(out, format="wav")
#             paths.append(out)
#         return paths
#     except Exception as e:
#         print("Error in chunk_audio:", e)
#         raise
#
#
# def transcribe_arabic(path):
#     try:
#         res = ASR_MODEL.transcribe(
#             path, task="transcribe", language="ar",
#             temperature=0.0, best_of=3
#         )
#         return res["text"].strip()
#     except Exception as e:
#         print(f"Error in transcribe_arabic for {path}:", e)
#         raise
#
#
# def segment_arabic_text(text, max_chars=200):
#     try:
#         parts = re.split(r'([.؟!])', text)
#         segments, buf = [], ""
#         for part in parts:
#             buf += part
#             if len(buf) >= max_chars or part in ".؟!":
#                 segments.append(buf.strip())
#                 buf = ""
#         if buf.strip():
#             segments.append(buf.strip())
#         return segments
#     except Exception as e:
#         print("Error in segment_arabic_text:", e)
#         raise
#
#
# def translate_arabic(ar_chunks):
#     translated = []
#     for idx, seg in enumerate(ar_chunks, start=1):
#         try:
#             batch = MT_TOKENIZER(seg, return_tensors="pt", truncation=True)
#             gen   = MT_MODEL.generate(**batch, max_length=128)
#             en    = MT_TOKENIZER.decode(gen[0], skip_special_tokens=True)
#             translated.append(f"({idx}) {seg}\n→ {en}")
#         except Exception as e:
#             print(f"Error translating segment {idx}:", e)
#             translated.append(f"({idx}) {seg}\n→ [translation error]")
#     return "\n\n".join(translated)
#
#
#
# @app.route('/upload-audio', methods=['POST'])
# def upload_audio():
#     # 1️⃣ File + user checks
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file    = request.files['file']
#     user_id = request.form.get('user_id')
#     if not file or not file.filename:
#         return jsonify({'error': 'No selected file'}), 400
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400
#
#     # 2️⃣ Save file
#     try:
#         filename = secure_filename(file.filename)
#         os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)
#     except Exception as e:
#         print("Error saving uploaded file:", e)
#         return jsonify({'error': 'Failed to save file'}), 500
#
#     # 3️⃣ Chunk audio
#     try:
#         chunks = chunk_audio(filepath, by_silence=True)
#         print("CHUNKS CCCCCCCCCCCCCCCCCCCC" ,chunks)
#     except Exception:
#         return jsonify({'error': 'Audio chunking failed'}), 500
#
#     # 4️⃣ Transcribe each chunk
#     full_ar = ""
#     for seg in chunks:
#         try:
#             text = transcribe_arabic(seg)
#             full_ar += text + " "
#         except Exception:
#             return jsonify({'error': f'Transcription failed on {seg}'}), 500
#
#     # 5️⃣ Segment Arabic text
#     try:
#         ar_segments = segment_arabic_text(full_ar, max_chars=250)
#     except Exception:
#         return jsonify({'error': 'Arabic text segmentation failed'}), 500
#
#     # 6️⃣ Translate each segment
#     try:
#         final_transcript = translate_arabic(ar_segments)
#         print("FINAL SUMMMARY ***************",final_transcript)
#     except Exception:
#         return jsonify({'error': 'Translation step failed'}), 500
#
#     # 7️⃣ (Optional) run NLP pipeline & save to DB …
#     #    summary = generate_summary(final_transcript)
#     #    keywords = extract_keywords(final_transcript)
#     #    …
#
#     return jsonify({
#         "message":    "Processed successfully",
#         "transcript": final_transcript,
#         # "summary":   summary,
#         # "keywords":  keywords,
#     }), 200
#
#
#
# def normalize_arabic(text: str) -> str:
#     """
#     1) Unicode NFC normalize
#     2) Strip all Arabic diacritics (harakat & tanwin)
#     3) Normalize hamza variants to bare alif
#     4) Collapse repeated spaces
#     """
#     # 1) NFC
#     text = unicodedata.normalize("NFC", text)
#
#     # 2) Remove tashkeel (diacritics)
#     arabic_diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]')
#     text = re.sub(arabic_diacritics, '', text)
#
#     # 3) Hamza normalization map
#     for src, tgt in {
#         'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
#         'ؤ': 'و', 'ئ': 'ي', 'ى': 'ي', 'ة': 'ه'
#     }.items():
#         text = text.replace(src, tgt)
#
#     # 4) Collapse whitespace
#     return re.sub(r'\s+', ' ', text).strip()
#
#
# def detect_quran_verses(text: str):
#     try:
#         normalized = normalize_arabic(text)
#         chunks = [chunk.strip() for chunk in normalized.split('\n') if chunk.strip()]
#         if len(chunks) == 1:
#             chunks = re.split(r'[،;؟\.\!]', normalized)
#             chunks = [c.strip() for c in chunks if len(c.strip()) > 3]
#
#         matches = []
#         seen = set()
#
#         for chunk in chunks:
#             if len(chunk) < 4 or len(chunk) > 200:
#                 continue
#
#             query = urllib.parse.quote(chunk)
#             url = f"https://api.alquran.cloud/v1/search/{query}/all/quran-uthmani"
#             try:
#                 res = requests.get(url, timeout=10)
#                 response = res.json()
#
#                 # Handle different response types
#                 data = response.get("data", {})
#
#                 # Check if data is a string (error message)
#                 if isinstance(data, str):
#                     app.logger.warning(f"Quran API message: {data}")
#                     continue
#
#                 # Check if we have valid matches data
#                 if isinstance(data, dict):
#                     matches_data = data.get("matches", [])
#                 else:
#                     matches_data = []
#
#                 for m in matches_data:
#                     key = (m["surah"]["englishName"], m["ayah"]["numberInSurah"])
#                     if key not in seen:
#                         seen.add(key)
#                         matches.append({
#                             "surah": m["surah"]["englishName"],
#                             "ayah_number": m["ayah"]["numberInSurah"],
#                             "text": m["ayah"]["text"]
#                         })
#             except requests.RequestException as e:
#                 app.logger.warning(f"Quran API request failed: {str(e)}")
#             except KeyError as e:
#                 app.logger.warning(f"Malformed Quran API response: {response}")
#         return matches
#
#     except Exception as e:
#         app.logger.error(f"Quran detection failed: {str(e)}")
#         return []
#
#
#
# def preprocess_transcript(text: str) -> str:
#     """
#     Preprocess Arabic/translated transcript:
#     - Remove diacritics (tashkeel)
#     - Normalize characters
#     - Remove non-Arabic symbols or English words (if needed)
#     - Fix elongations and repeated characters
#     """
#     # 1. Remove tashkeel/harakat (diacritics)
#     arabic_diacritics = re.compile(r'[\u064B-\u0652]')
#     text = re.sub(arabic_diacritics, '', text)
#
#     # 2. Normalize Arabic letters
#     replacements = {
#         'أ': 'ا',
#         'إ': 'ا',
#         'آ': 'ا',
#         'ى': 'ي',
#         'ئ': 'ي',
#         'ؤ': 'و',
#         'ة': 'ه',
#     }
#     for src, target in replacements.items():
#         text = text.replace(src, target)
#
#     # 3. Remove Latin characters, numbers, special symbols
#     text = re.sub(r'[a-zA-Z0-9]', '', text)
#     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
#
#     # 4. Remove excessive spaces
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     return text
#
#
# def detect_hadith(text: str) -> list[dict]:
#     """Detect hadith references in the text with API key validation"""
#     # Check if API key is available
#     if not HADITH_API_KEY or HADITH_API_KEY.strip() == "":
#         app.logger.error("Hadith API key is missing or empty")
#         return []
#
#     matches = []
#     seen = set()
#     headers = {'Authorization': f'Bearer {HADITH_API_KEY}'}
#
#     for line in lines:
#         try:
#             url = (
#                 "https://api.hadithapi.com/api/hadiths"
#                 f"?hadithEnglish={urllib.parse.quote(line)}"
#             )
#             resp = requests.get(url, headers=headers, timeout=5)
#             resp.raise_for_status()
#
#             data = resp.json().get("data", {})
#             for h in data.get("hadiths", []):
#                 key = (
#                     h.get("collection"),
#                     h.get("book"),
#                     h.get("reference", {}).get("hadithNumberInBook"),
#                 )
#                 if key not in seen:
#                     seen.add(key)
#                     matches.append({
#                         "collection": h.get("collection", ""),
#                         "book":       h.get("book", ""),
#                         "hadith_number": h.get("reference", {}).get("hadithNumberInBook", ""),
#                         "text":       h.get("hadithEnglish") or h.get("hadithArabic", "")
#                     })
#
#         except requests.exceptions.RequestException as e:
#             # network error, DNS failure, timeout, HTTP error, etc.
#             print(f"Hadith API request failed for line: {line!r} — {e}")
#             continue
#
#         except ValueError as e:
#             # JSON decode error
#             print(f"Hadith API returned invalid JSON for line: {line!r} — {e}")
#             continue
#
#     return matches
#
# # 1️⃣ Redirect user to Google
# @app.route('/login/google')
# def login_google():
#     redirect_uri = url_for('auth_google_callback', _external=True)
#     return google.authorize_redirect(redirect_uri)
#
#
# #
# # def detect_quran_verses(text: str):
# #     """
# #     Split the transcript into lines, search each via Al‑Quran.Cloud,
# #     and collect unique Surah/Ayah matches.
# #     """
# #     matches = []
# #     seen = set()
# #
# #     for line in [l.strip() for l in text.split('\n') if l.strip()]:
# #         # limit line length for reliable matching
# #         if len(line) < 5 or len(line) > 300:
# #             continue
# #         q = urllib.parse.quote(line)
# #         url = f"https://api.alquran.cloud/v1/search/{q}/all/quran-uthmani"
# #         print(url)
# #         try:
# #             r = requests.get(url, timeout=5)
# #             data = r.json().get("data", {})
# #             for m in data.get("matches", []):
# #                 key = (m["surah"]["englishName"], m["ayah"]["numberInSurah"])
# #                 if key not in seen:
# #                     seen.add(key)
# #                     matches.append({
# #                         "surah": m["surah"]["englishName"],
# #                         "ayah_number": m["ayah"]["numberInSurah"],
# #                         "text": m["ayah"]["text"]
# #                     })
# #         except:
# #             # connection or parse error—skip this line
# #             continue
# #
# #     return matches
# #
# # Semantic similarity using OpenAI
# # def semantic_match_with_openai(text, reference_type="quran"):
# #     prompt = f"""
# # You are an Islamic AI assistant.
# #
# # The following is a khutbah excerpt:
# # \"\"\"{text}\"\"\"
# #
# # Search for semantically matching {reference_type} reference(s) for this excerpt.
# # Return a JSON with:
# # - type: 'quran' or 'hadith'
# # - source: (e.g., "Surah Al-Baqarah, Ayah 2" or "Sahih Bukhari, Book 1, Hadith 1")
# # - text: the actual reference
# # - confidence: score between 0 to 1
# # """
# #     try:
# #         response = client.chat.completions.create(
# #             model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
# #             messages=[
# #                 {"role": "system", "content": "You are a helpful assistant knowledgeable in Quran and Hadith."},
# #                 {"role": "user", "content": prompt}
# #             ],
# #             temperature=0.3,
# #             max_tokens=500,
# #         )
# #         result = response.choices[0].message.content.strip()
# #         return json.loads(result)
# #     except Exception as e:
# #         print("❌ OpenAI semantic error:", e)
# #         return None
#
#
# # ---------------------------------------------------------
# # Semantic similarity using OpenAI
# # def semantic_match_with_openai(text, reference_type="quran"):
# #     prompt = f"""
# # You are an Islamic AI assistant.
# #
# # The following is a khutbah excerpt:
# # \"\"\"{text}\"\"\"
# #
# # Search for semantically matching {reference_type} reference(s) for this excerpt.
# # Return a JSON with:
# # - type: 'quran' or 'hadith'
# # - source: (e.g., "Surah Al-Baqarah, Ayah 2" or "Sahih Bukhari, Book 1, Hadith 1")
# # - text: the actual reference
# # - confidence: score between 0 to 1
# # """
# #     try:
# #         response = client.chat.completions.create(
# #             model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
# #             messages=[
# #                 {"role": "system", "content": "You are a helpful assistant knowledgeable in Quran and Hadith."},
# #                 {"role": "user", "content": prompt}
# #             ],
# #             temperature=0.3,
# #             max_tokens=500,
# #         )
# #         result = response.choices[0].message.content.strip()
# #         return json.loads(result)
# #     except Exception as e:
# #         print("❌ OpenAI semantic error:", e)
# #         return None
#
#
#
#
# # 2️⃣ Handle callback
# @app.route('/oauth2/callback/google')
# def auth_google_callback():
#     token = google.authorize_access_token()
#     userinfo = google.parse_id_token(token)
#     email = userinfo['email']
#     sub   = userinfo['sub']  # Google user ID
#
#     session = Session()
#     try:
#         # Check if user exists
#         user = session.query(User).filter_by(Email=email).first()
#         if not user:
#             # Create new user with Plan=demo
#             new_user = User(
#                 Id=uuid.uuid4(),
#                 Email=email,
#                 Password=sub,   # store Google sub as a placeholder
#                 Plan='demo',
#                 EmailConfirmed=True
#             )
#             session.add(new_user)
#             session.commit()
#             user = new_user
#
#         # Issue your own JWT or session cookie here
#         return jsonify({
#             'message': 'Google login successful',
#             'userId': str(user.Id),
#             'plan': user.Plan
#         }), 200
#
#     finally:
#         session.close()
#
#
# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.json
#     session = Session()
#     try:
#         # 1. Reject if email already in use (either confirmed or pending)
#         exists = (
#             session.query(User).filter_by(Email=data['email']).first()
#             or session.query(PendingUser).filter_by(Email=data['email']).first()
#         )
#         if exists:
#             return jsonify({'error': 'Email already registered or pending confirmation.'}), 400
#
#         # 2. Create a pending user
#         token = secrets.token_urlsafe(32)
#         pending_user = PendingUser(
#             Id=str(uuid.uuid4()),
#             Email=data['email'],
#             PasswordHash=generate_password_hash(data['password']),
#             ConfirmToken=token
#         )
#         session.add(pending_user)
#         session.commit()
#
#         # 3. Send confirmation email
#         confirm_url = f"http://192.168.18.97:5000/confirm-email?token={token}"
#         msg = Message("Confirm your email - Khutba.AI", recipients=[data['email']])
#         msg.body = (
#             f"Assalamu Alaikum,\n\n"
#             f"Please confirm your email by clicking the link below:\n{confirm_url}\n\n"
#             f"BarakAllah!"
#         )
#         mail.send(msg)
#
#         return jsonify({'message': 'Signup successful. Please confirm your email.'}), 201
#
#     except IntegrityError:
#         session.rollback()
#         return jsonify({'error': 'Email already exists'}), 400
#     except Exception as e:
#         session.rollback()
#         print("Signup Error:", e)  # helpful for debugging
#         return jsonify({'error': 'Internal server error'}), 500
#     finally:
#         session.close()
#
# @app.route('/confirm-email', methods=['GET'])
# def confirm_email():
#     token = request.args.get('token')
#     if not token:
#         return jsonify({'error': 'Missing token'}), 400
#
#     session = Session()
#     try:
#         pending = session.query(PendingUser).filter_by(ConfirmToken=token).first()
#         if not pending:
#             return jsonify({'error': 'Invalid or expired token'}), 400
#
#         user = User(
#             Id=pending.Id,
#             Email=pending.Email,
#             Password=pending.PasswordHash,
#             Plan='demo',
#             EmailConfirmed=True
#         )
#         session.add(user)
#         session.delete(pending)
#         session.commit()
#
#         return jsonify({'message': 'Email confirmed successfully! You can now log in.'}), 200
#
#     except Exception as e:
#         session.rollback()
#         print("Email confirmation error:", e)
#         return jsonify({'error': 'Internal server error'}), 500
#     finally:
#         session.close()
#
# # ---------------------------
# # 🔑 Login Route
# # ---------------------------
# @app.route('/login', methods=['POST'])
# def login():
#     data = request.json
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Email=data['email']).first()
#
#         if not user or not check_password_hash(user.Password, data['password']):
#             return jsonify({'error': 'Invalid credentials'}), 401
#
#         if not user.EmailConfirmed:
#             return jsonify({'error': 'Please confirm your email first'}), 403
#
#         return jsonify({
#             'message': 'Login successful',
#             'userId': str(user.Id),
#             'plan': user.Plan
#         }), 200
#
#     except Exception as e:
#         print("Login Error:", e)
#         return jsonify({'error': 'Internal server error'}), 500
#     finally:
#         session.close()
#
#
# # ---------------------------
# # ▶️ demo-khutbahs
# # ---------------------------
#
# @app.route('/demo-khutbahs', methods=['GET'])
# def demo_khutbahs():
#     khutbahs = [
#         {
#             "title": "🌟 Importance of Sincerity",
#             "summary": "This khutbah highlights the significance of Ikhlas (sincerity) in worship and daily life. It draws from Quran and Sunnah to show how intentions shape actions.",
#             "duration": "6 min",
#             "tags": ["ikhlas", "intentions", "worship"]
#         },
#         {
#             "title": "🤝 Unity in the Ummah",
#             "summary": "Explores the importance of staying united as an Ummah, resolving disputes peacefully, and building community ties as emphasized in Surah Al-Hujurat.",
#             "duration": "5 min",
#             "tags": ["unity", "brotherhood", "peace"]
#         },
#         {
#             "title": "🙏 Gratitude in Islam",
#             "summary": "Discusses how being thankful strengthens faith, improves mental well-being, and increases barakah, supported by verses from Surah Ibrahim and Hadith.",
#             "duration": "4 min",
#             "tags": ["gratitude", "barakah", "positivity"]
#         }
#     ]
#     return jsonify({"demo_khutbahs": khutbahs}), 200
#
# # ---------------------------
# # ▶️ Update from Demo to Premium
# # ---------------------------
# @app.route('/upgrade', methods=['POST'])
# def upgrade_to_premium():
#     data = request.json
#     user_id = data.get('user_id')
#
#     if not user_id:
#         return jsonify({'error': 'User ID required'}), 400
#
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         # Simulate successful payment and upgrade
#         user.Plan = 'premium'
#         session.commit()
#
#         return jsonify({'message': 'Plan upgraded to premium ✅'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶ Unsubscribe and convert from premium to demo
# # ---------------------------
# @app.route('/unsubscribe', methods=['POST'])
# def unsubscribe():
#     data = request.json
#     user_id = data.get('user_id')
#
#     if not user_id:
#         return jsonify({'error': 'User ID required'}), 400
#
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         if user.Plan == 'demo':
#             return jsonify({'message': 'Already unsubscribed (demo user).'}), 200
#
#         # Simulate unsubscribe
#         user.Plan = 'demo'
#         session.commit()
#
#         return jsonify({'message': 'Unsubscribed successfully. You are now on demo plan.'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
#
# # ---------------------------
# # ▶️ Helps frontend check user's current plan:
# # ---------------------------
# @app.route('/check-plan', methods=['GET'])
# def check_plan():
#     user_id = request.args.get('user_id')
#     session = Session()
#     user = session.query(User).filter_by(Id=user_id).first()
#     session.close()
#
#     if not user:
#         return jsonify({'error': 'User not found'}), 404
#
#     return jsonify({'plan': user.Plan}), 200
#
#
# # -------------------------------------------------------------------------------------------------
#
# # ---------------------------
# # ▶️ khutbah-archive
# # ---------------------------
#
# @app.route('/khutbah-archive', methods=['GET'])
# def khutbah_archive():
#     user_id = request.args.get('user_id')
#
#     if not user_id:
#         return jsonify({'error': 'User ID is required'}), 400
#
#     session = Session()
#     try:
#         khutbahs = session.query(Khutbah).filter_by(UserId=user_id).order_by(Khutbah.Created.desc()).all()
#
#         result = []
#         for k in khutbahs:
#             result.append({
#                 'id': str(k.Id),
#                 'audio_url': k.AudioUrl,
#                 'transcript': k.Transcript or "Not transcribed yet",
#                 'created': k.Created.strftime('%Y-%m-%d %H:%M:%S')
#             })
#
#         return jsonify({'khutbahs': result}), 200
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ get Khutba
# # ---------------------------
# @app.route('/get-khutbahs', methods=['GET'])
# def get_khutbahs():
#     user_id = request.args.get('user_id')
#     if not user_id:
#         return jsonify({'error': 'Missing user_id'}), 400
#
#     session = Session()
#     try:
#         # 🧑 Get user and check plan
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         # 🎯 Determine whose khutbahs to show
#         if user.Plan == 'demo':
#             # Fetch demo khutbahs tied to a fixed demo user ID
#             target_user_id = '00000000-0000-0000-0000-000000000001'
#         else:
#             # Fetch this user's khutbahs
#             target_user_id = user_id
#
#         khutbahs = (
#             session.query(Khutbah)
#             .filter_by(UserId=target_user_id)
#             .order_by(desc(Khutbah.Created))
#             .all()
#         )
#
#         results = []
#         for k in khutbahs:
#             try:
#                 sentiment_data = json.loads(k.Sentiment or '{}')
#             except:
#                 sentiment_data = {}
#
#             results.append({
#                 'id': str(k.Id),
#                 'audio_url': k.AudioUrl,
#                 'transcript': k.Transcript,
#                 'summary': k.Summary,
#                 'keywords': (k.Keywords or "").split(','),
#                 'sentiment': sentiment_data,
#                 'tips': k.Tips,
#                 'created': k.Created.strftime("%Y-%m-%d %H:%M")
#             })
#
#         return jsonify({'khutbahs': results}), 200
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#     finally:
#         session.close()
# # ---------------------------
# # ▶️ Search khutbahs
# # ---------------------------
# @app.route('/search-khutbahs', methods=['GET'])
# def search_khutbahs():
#     user_id = request.args.get('user_id')
#     query   = request.args.get('query', '').strip()
#
#     if not user_id or not query:
#         return jsonify({'error': 'user_id and query parameters are required'}), 400
#
#     session = Session()
#     try:
#         # Simple SQLAlchemy “LIKE” search across fields
#         pattern = f"%{query}%"
#         results = session.query(Khutbah) \
#             .filter(
#                 Khutbah.UserId == user_id,
#                 (
#                     Khutbah.Transcript.ilike(pattern) |
#                     Khutbah.Summary.ilike(pattern)    |
#                     Khutbah.Tags.ilike(pattern)
#                 )
#             ) \
#             .order_by(Khutbah.Created.desc()) \
#             .all()
#
#         khutbahs = []
#         for k in results:
#             khutbahs.append({
#                 'id': str(k.Id),
#                 'created': k.Created.strftime('%Y-%m-%d %H:%M:%S'),
#                 'summary': k.Summary or '',
#                 'tags': k.Tags.split(',') if k.Tags else [],
#                 'is_favorite': k.IsFavorite
#             })
#
#         return jsonify({'khutbahs': khutbahs}), 200
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
#
# # ---------------------------
# # ▶️ /khutbah/favorite
# # ---------------------------
#
# @app.route('/khutbah/favorite', methods=['POST'])
# def toggle_favorite():
#     data = request.json
#     user_id    = data.get('user_id')
#     khutbah_id = data.get('khutbah_id')
#
#     if not user_id or not khutbah_id:
#         return jsonify({'error': 'user_id and khutbah_id are required'}), 400
#
#     session = Session()
#     try:
#         khutbah = session.query(Khutbah) \
#             .filter_by(Id=khutbah_id, UserId=user_id) \
#             .first()
#         if not khutbah:
#             return jsonify({'error': 'Khutbah not found or access denied'}), 404
#
#         # Toggle
#         khutbah.IsFavorite = not khutbah.IsFavorite
#         session.commit()
#
#         return jsonify({
#             'id': khutbah_id,
#             'is_favorite': khutbah.IsFavorite
#         }), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Create-Checkout-session
# # ---------------------------
#
# @app.route('/create-checkout-session', methods=['POST'])
# def create_checkout_session():
#     data = request.json
#     user_id = data.get('user_id')
#
#     try:
#         session = stripe.checkout.Session.create(
#             payment_method_types=['card'],
#             line_items=[{
#                 'price_data': {
#                     'currency': 'usd',
#                     'unit_amount': 4000,  # $40 for Individual plan
#                     'product_data': {
#                         'name': 'Khutba.AI - Individual Plan',
#                     },
#                 },
#                 'quantity': 1,
#             }],
#             mode='payment',
#             success_url=f'http://localhost:5000/payment-success?user_id={user_id}',
#             cancel_url='http://localhost:5000/payment-cancelled',
#         )
#         return jsonify({'checkout_url': session.url})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# # ---------------------------
# # ▶️ Successful Payment
# # ---------------------------
# @app.route('/payment-success')
# def payment_success():
#     user_id = request.args.get('user_id')
#     session = Session()
#     user = session.query(User).filter_by(Id=user_id).first()
#     if user:
#         user.Plan = 'premium'
#         session.commit()
#     session.close()
#     return 'Payment successful! Your plan is now premium.'
# # ------------------------------------------------------------------------------
# # These track:
# # CurrentWeekGoal: how many reflections user wants to complete this week
# # WeeklyProgress: how many reflections they’ve done
# # Nurbits: earned reward points
# # LastGoalUpdate: helps reset goal every Saturday
#
# # ---------------------------------------------------------------------------------
#
# # ---------------------------
# # ▶️ set-weekly-goal
# # ---------------------------
# @app.route('/set-weekly-goal', methods=['POST'])
# def set_weekly_goal():
#     data = request.json
#     user_id = data.get('user_id')
#     goal = data.get('goal')
#
#     if not user_id or not goal:
#         return jsonify({'error': 'Missing user_id or goal'}), 400
#
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         user.CurrentWeekGoal = goal
#         user.WeeklyProgress = 0
#         user.LastGoalUpdate = datetime.utcnow()
#         session.commit()
#
#         return jsonify({'message': 'Weekly goal set successfully'}), 200
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#     finally:
#         session.close()
#
# # Increment Progress (triggered after reflection)
# def reward_progress(user_id):
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return
#
#         user.WeeklyProgress += 1
#         if user.WeeklyProgress >= user.CurrentWeekGoal:
#             user.Nurbits += 10  # 🟢 Earn reward
#
#         session.commit()
#     except:
#         session.rollback()
#     finally:
#         session.close()
#
# # Weekly Reset (every Saturday)
#
# def reset_weekly_goals():
#     session = Session()
#     try:
#         users = session.query(User).all()
#         for user in users:
#             if user.WeeklyProgress < user.CurrentWeekGoal:
#                 user.Nurbits = max(user.Nurbits - 5, 0)  # 🔴 Penalty
#
#             user.WeeklyProgress = 0
#             user.LastGoalUpdate = datetime.utcnow()
#         session.commit()
#     except:
#         session.rollback()
#     finally:
#         session.close()
#
#
#
# # ----------------------------------------------------------------------
# # Implement the Reflection Popup System and the Nurbit Star Logic.
#
# # 🎯 Purpose:
# # Force user to reflect on random khutbah summaries.
# # Lock screen for 20 seconds ⏱️ to encourage mindfulness.
# # When reflection is complete → reward Nurbits → update star.
#
# # ------------------------------------------------------------------------
#
# @app.route('/reflect', methods=['POST'])
# def reflect():
#     data = request.json
#     user_id = data.get('user_id')
#
#     if not user_id:
#         return jsonify({'error': 'Missing user_id'}), 400
#
#     session = Session()
#     try:
#         # 1️⃣ Get one random khutbah with summary
#         khutbah = (
#             session.query(Khutbah)
#             .filter(Khutbah.UserId == user_id, Khutbah.Summary.isnot(None))
#             .order_by(func.newid())  # RANDOM
#             .first()
#         )
#
#         if not khutbah:
#             return jsonify({'error': 'No khutbahs with summary found'}), 404
#
#         # 2️⃣ Reward Progress
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         user.WeeklyProgress += 1
#         earned = False
#         if user.WeeklyProgress >= user.CurrentWeekGoal:
#             user.Nurbits += 10
#             earned = True
#         session.commit()
#
#         return jsonify({
#             'summary': khutbah.Summary,
#             'timer': 20,
#             'nurbits': user.Nurbits,
#             'weekly_progress': user.WeeklyProgress,
#             'goal': user.CurrentWeekGoal,
#             'goal_reached': earned
#         }), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Test-reset-goals
# # ---------------------------
#
# @app.route('/test-reset-goals', methods=['POST'])
# def test_reset_goals():
#     from tasks import reset_weekly_goals
#     reset_weekly_goals()
#     return jsonify({'message': 'Test reset executed'}), 200
#
# # ---------------------------
# # ▶️ Add favourite to khutba
# # ---------------------------
#
# @app.route('/update-khutbah-meta', methods=['POST'])
# def update_khutbah_meta():
#     data = request.get_json()
#     khutbah_id = data.get('khutbah_id')
#     user_id = data.get('user_id')
#     is_favorite = data.get('is_favorite')
#     tags = data.get('tags')
#
#     if not khutbah_id or not user_id:
#         return jsonify({'error': 'Missing khutbah_id or user_id'}), 400
#
#     session = Session()
#     try:
#         khutbah = session.query(Khutbah).filter_by(Id=khutbah_id, UserId=user_id).first()
#         if not khutbah:
#             return jsonify({'error': 'Khutbah not found'}), 404
#
#         # Update only if values are provided
#         if is_favorite is not None:
#             khutbah.IsFavorite = bool(is_favorite)
#         if tags is not None:
#             khutbah.Tags = tags.strip()  # Comma-separated tags string
#
#         session.commit()
#         return jsonify({'message': 'Khutbah metadata updated successfully'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Filter-khutbahs
# # ---------------------------
#
# @app.route('/filter-khutbahs', methods=['GET'])
# def filter_khutbahs():
#     user_id = request.args.get('user_id')
#     tag = request.args.get('tag')
#     favorite = request.args.get('favorite')
#
#     if not user_id:
#         return jsonify({'error': 'Missing user_id'}), 400
#
#     session = Session()
#     try:
#         query = session.query(Khutbah).filter(Khutbah.UserId == user_id)
#
#         # Filter by favorite
#         if favorite is not None:
#             query = query.filter(Khutbah.IsFavorite == (favorite.lower() == 'true'))
#
#         # Filter by tag (partial match in comma-separated tags)
#         if tag:
#             query = query.filter(Khutbah.Tags.ilike(f'%{tag}%'))
#
#         khutbahs = query.order_by(Khutbah.Created.desc()).all()
#
#         result = []
#         for k in khutbahs:
#             result.append({
#                 "id": str(k.Id),
#                 "audio_url": k.AudioUrl,
#                 "transcript": k.Transcript,
#                 "summary": k.Summary,
#                 "keywords": k.Keywords.split(',') if k.Keywords else [],
#                 "sentiment": json.loads(k.Sentiment) if k.Sentiment else {},
#                 "tips": k.Tips,
#                 "created": k.Created.strftime("%Y-%m-%d %H:%M"),
#                 "is_favorite": k.IsFavorite,
#                 "tags": k.Tags
#             })
#
#         return jsonify({"khutbahs": result}), 200
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Edit khutba
# # ---------------------------
#
#
# @app.route('/edit-khutbah/<khutbah_id>', methods=['PUT'])
# def edit_khutbah(khutbah_id):
#     data = request.json
#     session = Session()
#     try:
#         khutbah = session.query(Khutbah).filter_by(Id=khutbah_id).first()
#         if not khutbah:
#             return jsonify({'error': 'Khutbah not found'}), 404
#
#         # Editable fields
#         if 'transcript' in data:
#             khutbah.Transcript = data['transcript']
#         if 'summary' in data:
#             khutbah.Summary = data['summary']
#         if 'tags' in data:
#             khutbah.Tags = data['tags']
#         if 'is_favorite' in data:
#             khutbah.IsFavorite = data['is_favorite']
#
#         session.commit()
#         return jsonify({'message': 'Khutbah updated successfully'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Delete Khutba
# # ---------------------------
#
# @app.route('/delete-khutbah/<khutbah_id>', methods=['DELETE'])
# def delete_khutbah(khutbah_id):
#     session = Session()
#     try:
#         khutbah = session.query(Khutbah).filter_by(Id=khutbah_id).first()
#         if not khutbah:
#             return jsonify({'error': 'Khutbah not found'}), 404
#
#         session.delete(khutbah)
#         session.commit()
#         return jsonify({'message': 'Khutbah deleted successfully'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#
#     finally:
#         session.close()
#
#
# # --------------------------------------------------------------------------------------
#           #-------------------- Profile Setting -----------------------------
# # ---------------------------
# # ▶️ Update Username and email
# # ---------------------------
# @app.route('/update-profile/<user_id>', methods=['PUT'])
# def update_profile(user_id):
#     data = request.json
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         # Update allowed fields
#         if 'username' in data:
#             user.Username = data['username']
#         if 'email' in data:
#             user.Email = data['email']
#
#         session.commit()
#         return jsonify({'message': 'Profile updated successfully'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#     finally:
#         session.close()
#
# # ---------------------------
# # ▶️ Change Password
# # ---------------------------
#
# @app.route('/change-password/<user_id>', methods=['PUT'])
# def change_password(user_id):
#     data = request.json
#     session = Session()
#     try:
#         user = session.query(User).filter_by(Id=user_id).first()
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#
#         old_password = data.get('old_password')
#         new_password = data.get('new_password')
#
#         if user.Password != old_password:
#             return jsonify({'error': 'Old password is incorrect'}), 403
#
#         user.Password = new_password
#         session.commit()
#         return jsonify({'message': 'Password changed successfully'}), 200
#
#     except Exception as e:
#         session.rollback()
#         return jsonify({'error': str(e)}), 500
#     finally:
#         session.close()
#
#
#
# # --------------------------------------------------------
# def check_apis():
#     results = {
#         'quran_api': {'status': 'down', 'response': None},
#         'hadith_api': {'status': 'down', 'response': None}
#     }
#
#     # 1. Quran API Check (Updated)
#     try:
#         quran_res = requests.get(
#             "https://api.alquran.cloud/v1/search/الله/all/ar.alafasy",
#             timeout=5
#         )
#         results['quran_api']['response'] = quran_res.json()
#
#         if quran_res.status_code == 200:
#             results['quran_api']['status'] = 'up'
#         else:
#             results['quran_api']['status'] = f'error ({quran_res.status_code})'
#
#     except Exception as e:
#         results['quran_api']['error'] = str(e)
#
#     # 2. Hadith API Check (Updated)
#     if HADITH_API_KEY:
#         try:
#             headers = {'Authorization': f'Bearer {HADITH_API_KEY}'}
#             hadith_res = requests.get(
#                 "https://hadithapi.com/api/hadiths?hadithArabic=الله",
#                 headers=headers,
#                 timeout=10
#             )
#             results['hadith_api']['response'] = hadith_res.json()
#
#             if hadith_res.status_code == 200:
#                 results['hadith_api']['status'] = 'up'
#             else:
#                 results['hadith_api']['status'] = f'error ({hadith_res.status_code})'
#
#         except Exception as e:
#             results['hadith_api']['error'] = str(e)
#
#     return results
#
#
#
# @app.route('/api-status',methods=['GET'])
# def api_status():
#     return jsonify(check_apis())
#
#
#
# # ---------------------------
# # ▶️ Run Server
# # ---------------------------
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
