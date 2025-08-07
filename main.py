import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import boto3

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Upload

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 setup
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")

# PostgreSQL setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create table(s)
Base.metadata.create_all(bind=engine)

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    foul_type: str = Form(...),
    notes: str = Form(None)
):
    # Generate unique key
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"videos/{timestamp}_{file.filename}"

    # Upload to S3
    s3_client.upload_fileobj(file.file, BUCKET_NAME, s3_key)
    s3_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

    # Save metadata to DB
    session = SessionLocal()
    try:
        upload = Upload(
            s3_url=s3_url,
            foul_type=foul_type,
            notes=notes
        )
        session.add(upload)
        session.commit()
    finally:
        session.close()

    return {
        "s3_url": s3_url,
        "foul_type": foul_type,
        "notes": notes,
        "status": "uploaded"
    }