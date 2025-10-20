import os
import pickle
import boto3
from config import S3_BUCKET, XGBOOST_MODEL_KEY, LABEL_ENCODER_KEY, AWS_REGION, MODEL_CACHE_DIR

# Cached models (loaded once)
_xgboost_model = None
_label_encoder = None


def load_pickle_from_s3(bucket, key, cache_name):
    """
    Download and load a pickle file from S3 with local caching
    """
    cache_path = os.path.join(MODEL_CACHE_DIR, cache_name)
    
    # Check if already cached locally
    if os.path.exists(cache_path):
        print(f"Loading {cache_name} from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Download from S3
    print(f"Downloading {cache_name} from S3...")
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # Create cache directory if doesn't exist
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    try:
        s3.download_file(bucket, key, cache_path)
        print(f"Successfully downloaded {cache_name}")
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


def get_xgboost_model():
    """
    Lazy load XGBoost model (downloads only once)
    """
    global _xgboost_model
    
    if _xgboost_model is None:
        _xgboost_model = load_pickle_from_s3(
            S3_BUCKET, 
            XGBOOST_MODEL_KEY, 
            "xgboost_model.pkl"
        )
        print("XGBoost model loaded successfully!")
    
    return _xgboost_model


def get_label_encoder():
    """
    Lazy load Label Encoder (downloads only once)
    """
    global _label_encoder
    
    if _label_encoder is None:
        _label_encoder = load_pickle_from_s3(
            S3_BUCKET, 
            LABEL_ENCODER_KEY, 
            "label_encoder.pkl"
        )
        print("Label encoder loaded successfully!")
    
    return _label_encoder