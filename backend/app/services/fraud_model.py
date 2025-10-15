"""
XGBoost Fraud Detection Model
- Load trained model and encoders
- Run fraud probability predictions
"""

import pickle
import pandas as pd
import numpy as np
import os

# Global variables for loaded model
MODEL = None
ENCODERS = None
FEATURE_COLUMNS = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'xgboost_fraud_model.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'label_encoders.pkl')


def load_fraud_model():
    """Load XGBoost model and label encoders on startup"""
    global MODEL, ENCODERS, FEATURE_COLUMNS
    
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"✅ Model loaded from: {MODEL_PATH}")
        
        # Load encoders
        with open(ENCODERS_PATH, 'rb') as f:
            ENCODERS = pickle.load(f)
        print(f"✅ Encoders loaded from: {ENCODERS_PATH}")
        
        # Define feature columns (same as training)
        FEATURE_COLUMNS = [
            'Transaction_Amount', 'Transaction_Type', 'Account_Balance',
            'Device_Type', 'Location', 'Merchant_Category', 'IP_Address_Flag',
            'Previous_Fraudulent_Activity', 'Daily_Transaction_Count',
            'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
            'Card_Type', 'Card_Age', 'Transaction_Distance',
            'Authentication_Method', 'Is_Weekend',
            'hour_of_day', 'day_of_week', 'is_unusual_hour',
            'amount_deviation_ratio', 'is_high_value', 'is_new_device'  # ✅ Correct order
        ]
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("   Make sure to copy model files to backend/artifacts/")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def predict_fraud_probability(transaction_data):
    """
    Predict fraud probability for a transaction
    
    Args:
        transaction_data: Dict with transaction details
    
    Returns:
        Float fraud probability (0.0 to 1.0)
    """
    global MODEL, ENCODERS, FEATURE_COLUMNS
    
    if MODEL is None:
        raise RuntimeError("Model not loaded. Call load_fraud_model() first.")
    
    try:
        # Prepare features
        features = prepare_features(transaction_data)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure correct column order
        features_df = features_df[FEATURE_COLUMNS]
        
        # Predict probability
        fraud_probability = MODEL.predict_proba(features_df)[0][1]  # Probability of class 1 (fraud)
        
        return float(fraud_probability)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Return moderate risk as fallback
        return 0.5


def prepare_features(transaction_data):
    """
    Prepare and engineer features for model input
    
    Args:
        transaction_data: Dict with raw transaction data
    
    Returns:
        Dict with all required features
    """
    from datetime import datetime
    from utils import get_user_baseline
    
    # Get user baseline for calculations
    user_baseline = get_user_baseline(transaction_data.get('user_id'))
    
    # Extract temporal features
    timestamp = datetime.now()
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_unusual_hour = 1 if 2 <= hour_of_day <= 6 else 0
    
    # Calculate derived features
    transaction_amount = transaction_data.get('transaction_amount', 0)
    avg_amount_7d = user_baseline.get('avg_transaction_amount', 1000)
    amount_deviation_ratio = transaction_amount / max(avg_amount_7d, 1)
    
    device_type = transaction_data.get('device_type', 'Desktop')
    known_devices = user_baseline.get('known_devices', [])
    is_new_device = 1 if device_type not in known_devices else 0
    
    highest_transaction = user_baseline.get('highest_transaction', 5000)
    is_high_value = 1 if transaction_amount > highest_transaction * 0.8 else 0
    
    # Encode categorical features
    features = {
        # Direct features
        'Transaction_Amount': transaction_amount,
        'Account_Balance': transaction_data.get('account_balance', 50000),
        'IP_Address_Flag': transaction_data.get('ip_address_flag', 0),
        'Previous_Fraudulent_Activity': user_baseline.get('fraud_history_count', 0),
        'Daily_Transaction_Count': transaction_data.get('daily_transaction_count', 1),
        'Avg_Transaction_Amount_7d': avg_amount_7d,
        'Failed_Transaction_Count_7d': transaction_data.get('failed_transaction_count_7d', 0),
        'Card_Age': transaction_data.get('card_age', 24),
        'Transaction_Distance': transaction_data.get('transaction_distance', 0),
        'Is_Weekend': is_weekend,
        
        # Temporal features
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_unusual_hour': is_unusual_hour,
        
        # Derived features
        'amount_deviation_ratio': amount_deviation_ratio,
        'is_new_device': is_new_device,
        'is_high_value': is_high_value,
        
        # Categorical features (will be encoded)
        'Transaction_Type': transaction_data.get('transaction_type', 'Online'),
        'Device_Type': device_type,
        'Location': transaction_data.get('location', 'Unknown'),
        'Merchant_Category': transaction_data.get('merchant_category', 'Retail'),
        'Card_Type': transaction_data.get('card_type', 'Credit'),
        'Authentication_Method': transaction_data.get('authentication_method', 'Password')
    }
    
    # Encode categorical variables
    categorical_features = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method']
    
    for col in categorical_features:
        if col in ENCODERS and col in features:
            try:
                # Encode using fitted encoder
                encoder = ENCODERS[col]
                value = features[col]
                
                # Handle unknown categories
                if value not in encoder.classes_:
                    # Use first class as default
                    value = encoder.classes_[0]
                
                features[col] = encoder.transform([value])[0]
            except Exception as e:
                print(f"⚠️  Encoding error for {col}: {e}")
                features[col] = 0
    
    return features