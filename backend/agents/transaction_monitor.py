import json
from datetime import datetime
from tools.ml_tools import predict_fraud
from tools.nova_tools import call_nova


def build_fraud_analysis_prompt(transaction_data, ml_result):
    """
    Build prompt for Nova to analyze fraud prediction
    """
    prompt = f"""You are an expert fraud detection analyst. Analyze this transaction and provide a comprehensive fraud assessment.

Transaction Details:
- Transaction ID: {transaction_data.get('transaction_id', 'N/A')}
- Amount: ${transaction_data.get('amount', 0)}
- Merchant Category: {transaction_data.get('merchant_category', 'Unknown')}
- Time: Hour {transaction_data.get('hour', 'N/A')}
- International: {transaction_data.get('is_international', False)}
- Card Present: {transaction_data.get('card_present', 'Unknown')}

ML Model Analysis:
- Fraud Probability: {ml_result['fraud_probability']} ({ml_result['fraud_score']}/100)
- Risk Level: {ml_result['risk_level']}
- Model Confidence: {ml_result['model_confidence']}

Based on the ML model results and transaction details, provide:

1. **Overall Verdict**: APPROVED, REJECTED, or REVIEW
2. **Risk Factors**: List 2-3 key risk indicators (or positive indicators if low risk)
3. **Recommended Action**: What should be done with this transaction?
4. **Reasoning**: Brief explanation (2-3 sentences) of your decision

Respond in the following JSON format:
{{
  "verdict": "APPROVED/REJECTED/REVIEW",
  "risk_factors": ["factor1", "factor2", "factor3"],
  "recommended_action": "action description",
  "reasoning": "explanation here"
}}

Provide ONLY the JSON response, no additional text."""

    return prompt


def analyze_transaction(transaction_data):
    """
    Main agent function: Analyze transaction using ML + Nova
    
    Args:
        transaction_data (dict): Transaction details
        
    Returns:
        dict: Complete fraud analysis
    """
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing Transaction: {transaction_data.get('transaction_id', 'Unknown')}")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Get ML prediction
        print("Step 1: Running XGBoost fraud detection...")
        ml_result = predict_fraud(transaction_data)
        print(f"✅ ML Prediction: {ml_result['risk_level']} risk ({ml_result['fraud_score']}/100)")
        
        # Step 2: Build prompt for Nova
        print("\nStep 2: Preparing analysis for Nova Pro...")
        prompt = build_fraud_analysis_prompt(transaction_data, ml_result)
        
        # Step 3: Get Nova's analysis
        print("Step 3: Consulting Nova Pro for expert analysis...")
        nova_response = call_nova(prompt, max_tokens=800, temperature=0.3)
        print("✅ Nova analysis complete")
        
        # Step 4: Parse Nova's JSON response
        try:
            nova_analysis = json.loads(nova_response)
        except json.JSONDecodeError:
            # Fallback if Nova doesn't return perfect JSON
            nova_analysis = {
                "verdict": "REVIEW",
                "risk_factors": ["Unable to parse Nova response"],
                "recommended_action": "Manual review required",
                "reasoning": nova_response
            }
        
        # Step 5: Combine results
        final_result = {
            "transaction_id": transaction_data.get("transaction_id"),
            "timestamp": datetime.now().isoformat(),
            "ml_prediction": ml_result,
            "nova_analysis": nova_analysis,
            "final_verdict": nova_analysis.get("verdict", "REVIEW"),
            "final_score": ml_result["fraud_score"]
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis Complete - Verdict: {final_result['final_verdict']}")
        print(f"{'='*60}\n")
        
        return final_result
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise