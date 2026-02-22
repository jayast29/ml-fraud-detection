import os
from dotenv import load_dotenv

load_dotenv()

def get_aws_config():
    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region": os.getenv("AWS_DEFAULT_REGION"),
        "bucket": os.getenv("S3_BUCKET")
    }