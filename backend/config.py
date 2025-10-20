import os

# S3 Model Configuration
S3_BUCKET = "aws-hackathon-models"  # CHANGE THIS
XGBOOST_MODEL_KEY = "artifacts/xgboost_fraud_model.pkl"  # CHANGE THIS
LABEL_ENCODER_KEY = "artifacts/label_encoders.pkl"  # CHANGE THIS

# AWS Configuration
AWS_REGION = "us-east-1"
NOVA_INFERENCE_ARN = "arn:aws:bedrock:us-east-1:427893119211:inference-profile/us.amazon.nova-pro-v1:0"

# Model cache directory
MODEL_CACHE_DIR = "/tmp/models"

# Replace FEATURE_COLUMNS with this (WITHOUT Risk_Score):
FEATURE_COLUMNS = [
    'Transaction_Amount',
    'Transaction_Type',
    'Account_Balance',
    'Device_Type',
    'Location',
    'Merchant_Category',
    'IP_Address_Flag',
    'Previous_Fraudulent_Activity',
    'Daily_Transaction_Count',
    'Avg_Transaction_Amount_7d',
    'Failed_Transaction_Count_7d',
    'Card_Type',
    'Card_Age',
    'Transaction_Distance',
    'Authentication_Method',
    'Is_Weekend',
    'hour_of_day',
    'day_of_week',
    'is_unusual_hour',
    'amount_deviation_ratio',
    'is_high_value',
    'is_new_device'
]

# Features Lambda provides
LAMBDA_FEATURES = [
    'transaction_amount',
    'transaction_type',
    'merchant_category',
    'card_type',
    'device_type',
    'location',
    'authentication_method',
    'user_id',
    'transaction_id'
]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    'Transaction_Type',
    'Device_Type',
    'Location',
    'Merchant_Category',
    'Card_Type',
    'Authentication_Method'
]

# Risk thresholds for feature generation
HIGH_VALUE_THRESHOLD = 1000.0  # Transactions above this are "high value"
UNUSUAL_HOURS = [0, 1, 2, 3, 4, 5, 23]  # Late night/early morning
MEDIAN_ACCOUNT_BALANCE = 50000.0  # Default account balance
MEDIAN_CARD_AGE = 730  # 2 years in days
MEDIAN_DAILY_TRANSACTIONS = 5
MEDIAN_AVG_TRANSACTION_7D = 200.0


# Deepfake Detection API
DEEPFAKE_API_ENDPOINT = "https://dmqgspb3oh.execute-api.us-east-1.amazonaws.com/prod/verify"  # UPDATE THIS

# DynamoDB Configuration
DYNAMODB_TABLE_NAME = "historical-transactions"  