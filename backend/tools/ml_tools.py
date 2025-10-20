import pandas as pd
import numpy as np
from tools.s3_tools import get_xgboost_model, get_label_encoder
from tools.feature_engineering import generate_all_features
from config import FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def prepare_features(transaction_data):
    """
    Prepare transaction data for XGBoost model
    
    Args:
        transaction_data (dict): Raw transaction data from Lambda
        
    Returns:
        pd.DataFrame: Prepared features in correct format
    """
    
    # Step 1: Generate all 23 features
    print("Generating features from Lambda input...")
    features = generate_all_features(transaction_data)
    
    # Step 2: Ensure features are in correct order
    ordered_features = {}
    for col in FEATURE_COLUMNS:
        if col in features:
            ordered_features[col] = features[col]
        else:
            print(f"Warning: Missing feature '{col}', using default 0")
            ordered_features[col] = 0
    
    # Step 3: Create DataFrame
    df = pd.DataFrame([ordered_features])
    
    # Step 4: Encode categorical columns
    encoders = get_label_encoder()  # This is a dict of encoders
    
    # Check if it's a dict or single encoder
    is_dict = isinstance(encoders, dict)
    
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            try:
                original_value = df[col].iloc[0]
                
                if is_dict:
                    # Multiple encoders (one per column)
                    if col in encoders:
                        encoder = encoders[col]
                        if original_value in encoder.classes_:
                            df[col] = encoder.transform([original_value])[0]
                        else:
                            print(f"Warning: Unknown '{original_value}' in {col}, using 0")
                            df[col] = 0
                    else:
                        print(f"Warning: No encoder for {col}, using 0")
                        df[col] = 0
                else:
                    # Single encoder (your old approach)
                    if original_value in encoders.classes_:
                        df[col] = encoders.transform([original_value])[0]
                    else:
                        print(f"Warning: Unknown '{original_value}' in {col}, using 0")
                        df[col] = 0
                    
            except Exception as e:
                print(f"Error encoding {col}: {e}")
                df[col] = 0
    
    print(f"✅ Features prepared: {df.shape}")
    return df

def predict_fraud(transaction_data):
    """
    Predict fraud probability using XGBoost model
    
    Args:
        transaction_data (dict): Transaction details from Lambda
        
    Returns:
        dict: Fraud prediction results
    """
    try:
        # Load model
        model = get_xgboost_model()
        
        # Prepare features
        features = prepare_features(transaction_data)
        
        # Get prediction probability
        fraud_prob = model.predict_proba(features)[0][1]  # Probability of fraud (class 1)
        
        # Calculate fraud score (0-100)
        fraud_score = int(fraud_prob * 100)
        
        # Determine risk level
        if fraud_score < 30:
            risk_level = "LOW"
        elif fraud_score < 70:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Get model confidence (max probability)
        confidence = max(model.predict_proba(features)[0])
        
        # Get feature values for transparency (safely access features)
        feature_summary = {
            'amount': transaction_data.get('transaction_amount'),
            'type': transaction_data.get('transaction_type'),
            'merchant': transaction_data.get('merchant_category'),
            'time_based_risk': int(features['is_unusual_hour'].iloc[0]) if 'is_unusual_hour' in features.columns else 0,
            'hour': int(features['hour_of_day'].iloc[0]) if 'hour_of_day' in features.columns else 0,
            'is_high_value': int(features['is_high_value'].iloc[0]) if 'is_high_value' in features.columns else 0
        }
        
        return {
            "fraud_probability": round(fraud_prob, 4),
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "model_confidence": round(confidence, 4),
            "feature_summary": feature_summary
        }
        
    except Exception as e:
        print(f"Error in fraud prediction: {e}")
        import traceback
        traceback.print_exc()
        raise