from sqlalchemy import Column, String, UnicodeText, Unicode,DateTime, Text, ForeignKey, Integer, Date
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from datetime import datetime
import uuid

from config import Base

from sqlalchemy import Column, String, Boolean

class User(Base):
    __tablename__ = 'Users'

    Id = Column(String(36), primary_key=True)
    Email = Column(String(100), unique=True, nullable=False)
    Password = Column(String(100), nullable=False)
    Plan = Column(String(20), default='demo')
    EmailConfirmed = Column(Boolean, default=False)
    ConfirmToken = Column(String(100), nullable=True)
    # 🔻 Add these missing fields
    WeeklyProgress = Column(Integer, default=0)
    CurrentWeekGoal = Column(Integer, default=5)
    Nurbits = Column(Integer, default=0)
    EmailConfirmed = Column(Boolean, default=False)
    ConfirmToken = Column(String(255), nullable=True)


class Khutbah(Base):
    __tablename__ = 'Khutbahs'

    Id         = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    UserId     = Column(UNIQUEIDENTIFIER, ForeignKey('Users.Id'), nullable=False)
    AudioUrl   = Column(Unicode(500), nullable=False)
    Transcript = Column(UnicodeText)       # ← UnicodeText
    Summary    = Column(UnicodeText)       # ← UnicodeText
    Keywords   = Column(Unicode(500))      # ← Unicode
    Sentiment  = Column(UnicodeText)
    Tips       = Column(UnicodeText)
    Tags       = Column(Unicode(500))
    IsFavorite = Column(Boolean, default=False)
    Created    = Column(DateTime, default=datetime.utcnow)
    # models.py (add inside your Khutbah class)
    Verses = Column(Text, nullable=True)  # Can store as JSON string


class Intention(Base):
    __tablename__ = 'Intentions'

    Id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    UserId = Column(UNIQUEIDENTIFIER, ForeignKey('Users.Id'), nullable=False)
    WeekStart = Column(Date, nullable=False)
    Intention = Column(String(500))
    Level = Column(Integer, nullable=False)


