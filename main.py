import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import boto3

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS S3 configuration
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
BUCKET_NAME = os.getenv("AWS_S3_BUCKET")

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),          # ✅ Correct parameter name
    foul_type: str = Form(...),
    notes: str = Form(None)
):
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"videos/{timestamp}_{file.filename}"

    # Upload to S3
    s3_client.upload_fileobj(file.file, BUCKET_NAME, s3_key)

    # Construct public S3 URL
    s3_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

    print(f"✅ Uploaded to S3: {s3_url}")
    print(f"Foul Type: {foul_type}")
    print(f"Notes: {notes}")

    return {
        "s3_url": s3_url,
        "foul_type": foul_type,
        "notes": notes,
        "status": "uploaded"
    }