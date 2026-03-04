from sqlalchemy import Column, Integer, String, Text, Float, MetaData, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from app.core.config import get_settings

settings = get_settings()
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Sentence(Base):
    __tablename__ = "sentences"

    id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), index=True)
    chapter_number = Column(Integer, index=True)
    chapter_title = Column(String(255))
    sentence_index = Column(Integer)
    sentence_text = Column(Text)
    source_type = Column(String(20), default="summary")
    embedding = Column(Vector(768))

class Theme(Base):
    __tablename__ = "themes"

    id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), index=True)
    tagalog_title = Column(String(255))
    meaning = Column(Text)
    embedding = Column(Vector(768))
