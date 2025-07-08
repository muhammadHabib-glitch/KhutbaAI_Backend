from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- Database Config ----------
Base = declarative_base()

DATABASE_URI = r"mssql+pyodbc://sa:habibfarooq12345@DESKTOP-8TUN3M3\SQLEXPRESS/KhutbaAI?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DATABASE_URI, echo=True)
Session = sessionmaker(bind=engine)





# Get Hadith API key
HADITH_API_KEY = os.getenv("HADITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# -------------------------------------------------------------------
STRIPE_SECRET_KEY = "sk_test_..."  # from your Stripe dashboard


# ---------- Mail Config ----------
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USE_TLS = True
MAIL_USERNAME = 'habibfarooq25@gmail.com'           # 👈 Replace with your Gmail
MAIL_PASSWORD = 'blqh llfq pxnq ipzy'         # 👈 Replace with Gmail App Password
MAIL_DEFAULT_SENDER = MAIL_USERNAME


# config.py (add these at the bottom)

#----------- Google OAuth2 --------------
GOOGLE_CLIENT_ID     = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "YOUR_GOOGLE_CLIENT_SECRET"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# A random 32‑byte secret; in production, set this via an environment variable
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))