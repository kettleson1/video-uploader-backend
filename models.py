from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    s3_url = Column(String, nullable=False)
    foul_type = Column(String, nullable=False)
    notes = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)