"""
AWS Services Integration - LOCAL DEVELOPMENT VERSION
Only using Bedrock Claude, rest are mocked
"""

import boto3
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Get credentials from .env
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION', 'us-east-1')

# Initialize Bedrock client with explicit credentials
bedrock_client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region_name
)

# Mock data stores (in-memory for local dev)
MOCK_USER_PROFILES = {}
MOCK_TRANSACTIONS = {}
MOCK_FRAUD_CASES = {}


# =============================================================================
# BEDROCK - Claude LLM (REAL AWS CALL)
# =============================================================================

def call_bedrock_claude(prompt, max_tokens=500):
    """Call Bedrock Claude for LLM reasoning"""
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_client.invoke_model(
            modelId=os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-4-20250514'),
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    except Exception as e:
        print(f"❌ Bedrock error: {str(e)}")
        return f"Error generating explanation: {str(e)}"


# =============================================================================
# MOCK FUNCTIONS - Local In-Memory Storage
# =============================================================================

def get_user_profile(user_id):
    """Get user profile (MOCK - returns hardcoded data)"""
    print(f"   🔧 [MOCK] Getting user profile: {user_id}")
    
    # Return mock data
    return MOCK_USER_PROFILES.get(user_id, {
        'user_id': user_id,
        'avg_transaction_amount': 1000,
        'known_devices': ['Desktop', 'Mobile'],
        'account_age_days': 365,
        'total_transactions': 150
    })


def get_user_transaction_history(user_id, days=90):
    """Get transaction history (MOCK - returns empty list)"""
    print(f"   🔧 [MOCK] Getting transaction history: {user_id} (last {days} days)")
    
    # Return mock history
    return [
        {
            'transaction_id': 'TXN_001',
            'timestamp': (datetime.now() - timedelta(hours=5)).isoformat(),
            'transaction_amount': 850,
            'merchant_category': 'Retail'
        },
        {
            'transaction_id': 'TXN_002',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'transaction_amount': 1200,
            'merchant_category': 'Food'
        }
    ]


def save_pending_transaction(transaction_id, transaction_data, verification_code, flag_reasons):
    """Save pending transaction (MOCK - stores in memory)"""
    print(f"   🔧 [MOCK] Saving pending transaction: {transaction_id}")
    
    MOCK_TRANSACTIONS[transaction_id] = {
        **transaction_data,
        'status': 'PENDING',
        'verification_code': verification_code,
        'flag_reasons': flag_reasons,
        'created_at': datetime.now().isoformat()
    }


def get_pending_transaction(transaction_id):
    """Retrieve pending transaction (MOCK - from memory)"""
    print(f"   🔧 [MOCK] Retrieving transaction: {transaction_id}")
    return MOCK_TRANSACTIONS.get(transaction_id)


def save_uploaded_photo(transaction_id, photo_data, filename):
    """Save photo (MOCK - returns fake S3 path)"""
    print(f"   🔧 [MOCK] Saving photo: {filename}")
    return f"s3://mock-bucket/uploads/{transaction_id}_{filename}"


def get_reference_photo(user_id):
    """Get reference photo (MOCK - returns fake S3 path)"""
    print(f"   🔧 [MOCK] Getting reference photo: {user_id}")
    return f"s3://mock-bucket/reference/{user_id}.jpg"


def invoke_sagemaker_endpoint(input_data):
    """Call SageMaker (MOCK - returns fake deepfake result)"""
    print(f"   🔧 [MOCK] Calling SageMaker deepfake detector")
    
    # Return mock deepfake detection result
    return {
        'is_deepfake': False,
        'deepfake_confidence': 0.12,
        'face_match_score': 0.95,
        'code_validated': True
    }


def create_fraud_case(fraud_case):
    """Create fraud case (MOCK - stores in memory)"""
    print(f"   🔧 [MOCK] Creating fraud case: {fraud_case['case_id']}")
    MOCK_FRAUD_CASES[fraud_case['case_id']] = fraud_case


def save_evidence_package(case_id, evidence_package):
    """Save evidence (MOCK - returns fake S3 path)"""
    print(f"   🔧 [MOCK] Saving evidence package: {case_id}")
    return f"s3://mock-bucket/evidence/{case_id}/evidence.json"


def send_sns_alert(subject, message):
    """Send alert (MOCK - just prints)"""
    print(f"   🔧 [MOCK] Sending SNS alert: {subject}")
    print(f"   📧 {message[:100]}...")
    return {'MessageId': 'mock-message-id'}