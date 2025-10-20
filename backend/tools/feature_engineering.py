from datetime import datetime
import numpy as np
from config import (
    HIGH_VALUE_THRESHOLD, 
    UNUSUAL_HOURS,
    MEDIAN_ACCOUNT_BALANCE,
    MEDIAN_CARD_AGE,
    MEDIAN_DAILY_TRANSACTIONS,
    MEDIAN_AVG_TRANSACTION_7D
)


def generate_time_features(timestamp=None):
    """
    Generate time-based features from timestamp
    
    Args:
        timestamp: datetime object or ISO string or None (uses current time)
        
    Returns:
        dict: Time-based features
    """
    if timestamp is None:
        dt = datetime.now()
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            dt = datetime.now()
    else:
        dt = timestamp
    
    return {
        'hour_of_day': dt.hour,
        'day_of_week': dt.weekday(),  # 0=Monday, 6=Sunday
        'Is_Weekend': 1 if dt.weekday() >= 5 else 0,
        'is_unusual_hour': 1 if dt.hour in UNUSUAL_HOURS else 0
    }


def calculate_risk_score(transaction_data, time_features):
    """
    Calculate a heuristic risk score based on available data
    
    Returns score between 0 and 1
    """
    risk_score = 0.0
    
    amount = transaction_data.get('transaction_amount', 0)
    
    # Factor 1: Transaction amount (0-0.3)
    if amount > 5000:
        risk_score += 0.3
    elif amount > 2000:
        risk_score += 0.2
    elif amount > 1000:
        risk_score += 0.1
    elif amount < 10:
        risk_score += 0.05  # Very small amounts can be testing
    
    # Factor 2: Time of day (0-0.2)
    if time_features['is_unusual_hour']:
        risk_score += 0.2
    
    # Factor 3: Transaction type (0-0.15)
    txn_type = transaction_data.get('transaction_type', '').lower()
    if txn_type in ['withdrawal', 'transfer']:
        risk_score += 0.15
    elif txn_type == 'purchase':
        risk_score += 0.05
    
    # Factor 4: Merchant category (0-0.15)
    merchant = transaction_data.get('merchant_category', '').lower()
    high_risk_merchants = ['electronics', 'jewelry', 'travel', 'crypto']
    if any(hrm in merchant for hrm in high_risk_merchants):
        risk_score += 0.15
    
    # Factor 5: Location (0-0.1)
    location = transaction_data.get('location', '').lower()
    if 'unknown' in location or 'international' in location:
        risk_score += 0.1
    
    # Factor 6: Authentication (0-0.1)
    auth = transaction_data.get('authentication_method', '').lower()
    if auth in ['none', 'password']:
        risk_score += 0.1
    
    return min(risk_score, 1.0)  # Cap at 1.0


def calculate_amount_deviation(amount, avg_amount_7d):
    """
    Calculate how much this transaction deviates from average
    """
    if avg_amount_7d == 0:
        return 0.0
    return abs(amount - avg_amount_7d) / avg_amount_7d


def generate_all_features(transaction_data):
    """
    Generate ALL features from Lambda input
    """
    
    # Step 1: Get time features
    timestamp = transaction_data.get('timestamp')
    time_features = generate_time_features(timestamp)
    
    # Step 2: Map Lambda names to model names (capitalize first letter)
    amount = transaction_data.get('transaction_amount', 0)
    
    # Step 3: Generate user history features (defaults for now)
    avg_amount_7d = MEDIAN_AVG_TRANSACTION_7D
    daily_txn_count = MEDIAN_DAILY_TRANSACTIONS
    
    # Step 4: Calculate derived features
    # Calculate risk_score for internal use, but DON'T add to features
    # risk_score = calculate_risk_score(transaction_data, time_features)
    amount_deviation = calculate_amount_deviation(amount, avg_amount_7d)
    
    # Step 5: Build complete feature dictionary (WITHOUT Risk_Score)
    features = {
        # Direct from Lambda (renamed)
        'Transaction_Amount': amount,
        'Transaction_Type': transaction_data.get('transaction_type', 'Purchase'),
        'Device_Type': transaction_data.get('device_type', 'Desktop'),
        'Location': transaction_data.get('location', 'Domestic'),
        'Merchant_Category': transaction_data.get('merchant_category', 'Retail'),
        'Card_Type': transaction_data.get('card_type', 'Credit'),
        'Authentication_Method': transaction_data.get('authentication_method', 'PIN'),
        
        # User history features (defaults)
        'Account_Balance': MEDIAN_ACCOUNT_BALANCE,
        'Previous_Fraudulent_Activity': 0,
        'Daily_Transaction_Count': daily_txn_count,
        'Avg_Transaction_Amount_7d': avg_amount_7d,
        'Failed_Transaction_Count_7d': 0,
        'Card_Age': MEDIAN_CARD_AGE,
        
        # Location/network features
        'IP_Address_Flag': 0,
        'Transaction_Distance': 0,
        'is_new_device': 0,
        
        # Calculated features (NO Risk_Score)
        'amount_deviation_ratio': amount_deviation,
        'is_high_value': 1 if amount > HIGH_VALUE_THRESHOLD else 0,
        
        # Time features
        'hour_of_day': time_features['hour_of_day'],
        'day_of_week': time_features['day_of_week'],
        'Is_Weekend': time_features['Is_Weekend'],
        'is_unusual_hour': time_features['is_unusual_hour']
    }
    
    return features

def enrich_transaction_with_user_history(features, user_id, dynamodb_table=None):
    """
    OPTIONAL: Enrich features with real user history from DynamoDB
    
    Call this function if you have DynamoDB setup with user transaction history
    
    Args:
        features (dict): Feature dictionary from generate_all_features()
        user_id (str): User identifier
        dynamodb_table: Boto3 DynamoDB table resource
        
    Returns:
        dict: Updated features with real user history
    """
    
    if dynamodb_table is None:
        # No DynamoDB, return as-is
        return features
    
    try:
        # Query user's recent transactions
        # This is a placeholder - implement based on your DynamoDB schema
        
        # Example query logic:
        # response = dynamodb_table.query(
        #     KeyConditionExpression='user_id = :uid',
        #     ExpressionAttributeValues={':uid': user_id}
        # )
        
        # Calculate from real history:
        # features['Daily_Transaction_Count'] = calculate_daily_count(response)
        # features['Avg_Transaction_Amount_7d'] = calculate_avg_7d(response)
        # features['Failed_Transaction_Count_7d'] = count_failed(response)
        # features['Previous_Fraudulent_Activity'] = check_fraud_history(response)
        
        pass
    
    except Exception as e:
        print(f"Error fetching user history: {e}")
        # Keep default values
    
    return features