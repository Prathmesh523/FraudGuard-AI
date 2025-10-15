"""
Agent 1: Transaction Monitor
- Runs XGBoost model for fraud prediction
- Uses Bedrock Claude for natural language explanation
"""

import sys
sys.path.append('..')

from services.fraud_model import predict_fraud_probability
from services.aws_services import call_bedrock_claude
from utils import extract_features_for_ml


def analyze_transaction(transaction_data):
    """
    Analyze transaction using ML model + LLM explanation
    
    Args:
        transaction_data: Dict with transaction details
    
    Returns:
        Dict with fraud probability, risk score, and explanation
    """
    
    print("   🔍 Running XGBoost fraud detection...")
    
    # =========================================================================
    # Step 1: Get ML Prediction
    # =========================================================================
    fraud_probability = predict_fraud_probability(transaction_data)
    risk_score = int(fraud_probability * 100)
    
    print(f"   📊 Fraud Probability: {fraud_probability:.2%}")
    print(f"   🎯 Risk Score: {risk_score}/100")
    
    # =========================================================================
    # Step 2: Get LLM Explanation from Claude
    # =========================================================================
    print("   🤖 Generating explanation via Bedrock Claude...")
    
    # Build prompt for Claude
    prompt = f"""You are a senior fraud analyst. Analyze this transaction and explain the fraud risk.

Transaction Details:
- Amount: ${transaction_data.get('transaction_amount', 0):,.2f}
- Type: {transaction_data.get('transaction_type', 'Unknown')}
- Merchant: {transaction_data.get('merchant_category', 'Unknown')}
- Device: {transaction_data.get('device_type', 'Unknown')}
- Location: {transaction_data.get('location', 'Unknown')}
- User ID: {transaction_data.get('user_id', 'Unknown')}

ML Model Fraud Probability: {fraud_probability:.2%}
Risk Score: {risk_score}/100

Provide a concise 2-3 sentence explanation of why this transaction received this risk score. Focus on key risk indicators.
"""
    
    # Call Bedrock Claude
    llm_explanation = call_bedrock_claude(prompt)
    
    print(f"   ✅ Explanation: {llm_explanation}...")
    
    # =========================================================================
    # Step 3: Determine Recommendation
    # =========================================================================
    if risk_score >= 85:
        recommendation = "BLOCK"
    elif risk_score >= 60:
        recommendation = "REVIEW"
    else:
        recommendation = "APPROVE"
    
    return {
        'fraud_probability': fraud_probability,
        'risk_score': risk_score,
        'llm_explanation': llm_explanation,
        'recommendation': recommendation,
        'agent': 'transaction_monitor'
    }