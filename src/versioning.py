import os
import subprocess
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import get_aws_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running preprocessing...")
    
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df.drop(columns=['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], inplace=True)
    
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    df.drop(columns=['nameOrig', 'nameDest'], inplace=True)
    df.fillna(0, inplace=True)
    
    logger.info("Preprocessing complete")
    return df

def create_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('isFraud', group_keys=False).apply(lambda x: x.sample(frac=0.01, random_state=42))

def add_to_dvc(files: list):
    for f in files:
        logger.info(f"Adding {f} to DVC...")
        subprocess.run(["dvc", "add", f], check=True, cwd="..")

def push_to_s3():
    config = get_aws_config()
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = config["key"]
    env["AWS_SECRET_ACCESS_KEY"] = config["secret"]
    env["AWS_DEFAULT_REGION"] = config["region"]
    
    logger.info(f"Pushing to bucket: {config['bucket']}")
    subprocess.run(["dvc", "push"], check=True, env=env, cwd="..")
    logger.info("Push complete")

if __name__ == "__main__":
    logger.info("Loading raw data...")
    df = pd.read_csv("data/raw_data.csv")
    
    df = preprocess(df)
    
    df.to_csv("data/data.csv", index=False)
    logger.info("Saved data.csv")