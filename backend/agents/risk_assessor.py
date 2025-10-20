import json
from datetime import datetime
from tools.nova_tools import call_nova


def build_risk_assessment_prompt(transaction_data, fraud_result, evidence_result, deepfake_result):
    """
    Build comprehensive prompt for risk assessment
    """
    
    # Extract key metrics
    ml_score = fraud_result.get('final_score', 0)
    ml_verdict = fraud_result.get('final_verdict', 'UNKNOWN')
    ml_risk_level = fraud_result.get('ml_prediction', {}).get('risk_level', 'UNKNOWN')
    
    evidence_patterns = evidence_result.get('detected_patterns', [])
    evidence_summary = evidence_result.get('llm_summary', 'Not available')
    user_profile = evidence_result.get('user_profile', {})
    
    has_photo = deepfake_result is not None and not deepfake_result.get('error')
    biometric_verdict = deepfake_result.get('final_verdict', 'NOT_CHECKED') if deepfake_result else 'NOT_CHECKED'
    biometric_authentic = deepfake_result.get('verification_summary', {}).get('is_authentic', False) if has_photo else None
    
    prompt = f"""You are a senior fraud risk analyst. You have received analysis from three specialized AI agents. Your job is to synthesize ALL findings into a comprehensive, detailed fraud risk assessment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 TRANSACTION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transaction ID: {transaction_data.get('transaction_id')}
User ID: {transaction_data.get('user_id')}
Amount: ${transaction_data.get('transaction_amount', 0):,.2f}
Type: {transaction_data.get('transaction_type')}
Merchant: {transaction_data.get('merchant_category')}
Location: {transaction_data.get('location')}
Device: {transaction_data.get('device_type')}
Card: {transaction_data.get('card_type')}
Authentication: {transaction_data.get('authentication_method')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 AGENT 1: ML FRAUD DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verdict: {ml_verdict}
Fraud Score: {ml_score}/100
Risk Level: {ml_risk_level}
Model Confidence: {fraud_result.get('ml_prediction', {}).get('model_confidence', 0):.2%}

Key Indicators:
- Transaction Amount: ${transaction_data.get('transaction_amount', 0):,.2f}
- Time-Based Risk: {"HIGH" if fraud_result.get('ml_prediction', {}).get('feature_summary', {}).get('time_based_risk') else "NORMAL"}
- High Value Flag: {"YES" if fraud_result.get('ml_prediction', {}).get('feature_summary', {}).get('is_high_value') else "NO"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 AGENT 2: EVIDENCE & BEHAVIORAL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User Profile:
- Total Transactions: {user_profile.get('total_transactions', 0)}
- Average Transaction: ${user_profile.get('avg_transaction_amount', 0):,.2f}
- Transaction Range: ${user_profile.get('transaction_range', {}).get('min', 0):,.2f} - ${user_profile.get('transaction_range', {}).get('max', 0):,.2f}
- Known Devices: {', '.join(user_profile.get('known_devices', [])[:3])}
- Known Locations: {', '.join(user_profile.get('known_locations', [])[:3])}
- Fraud History: {user_profile.get('fraud_history', 0)} previous incidents

Suspicious Patterns Detected ({len(evidence_patterns)}):
{chr(10).join(['  ' + p for p in evidence_patterns]) if evidence_patterns else '  None detected'}

Behavioral Summary:
{evidence_summary}

Recent Activity:
- Last 24 hours: {evidence_result.get('transaction_history', {}).get('recent_24h_count', 0)} transactions
- Historical (90d): {evidence_result.get('transaction_history', {}).get('total_count', 0)} transactions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎭 AGENT 3: BIOMETRIC VERIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: {biometric_verdict}
Photo Provided: {"YES" if has_photo else "NO"}
{f'''
Verification Results:
- Authentic Identity: {"YES" if biometric_authentic else "NO - POTENTIAL FRAUD"}
- Face Match: {deepfake_result.get('detailed_checks', {}).get('face_match', {}).get('similarity', 0):.1f}% similarity
- Deepfake Check: {"PASSED" if deepfake_result.get('detailed_checks', {}).get('deepfake_detection', {}).get('is_real') else "FAILED - DEEPFAKE DETECTED"}
- Liveness Check: {"PASSED" if deepfake_result.get('detailed_checks', {}).get('liveness_check', {}).get('is_live') else "FAILED - NOT LIVE PERSON"}
- Overall Confidence: {deepfake_result.get('verification_summary', {}).get('overall_confidence', 0):.1f}%

Risk Factors: {', '.join(deepfake_result.get('detailed_checks', {}).get('risk_factors', [])) if deepfake_result.get('detailed_checks', {}).get('risk_factors') else 'None'}
''' if has_photo else 'Photo verification was not performed for this transaction.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR COMPREHENSIVE RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on ALL the evidence above, provide a detailed risk assessment in JSON format:

{{
  "final_verdict": "APPROVED/REJECTED/REVIEW",
  "confidence_score": 85,
  "risk_level": "LOW/MEDIUM/HIGH/CRITICAL",
  
  "risk_summary": "2-3 sentence executive summary of the overall risk",
  
  "key_findings": [
    "Finding 1: Most critical issue or positive indicator",
    "Finding 2: Second most important factor",
    "Finding 3: Additional relevant observation"
  ],
  
  "risk_factors": [
    "Specific risk factor 1",
    "Specific risk factor 2"
  ],
  
  "positive_indicators": [
    "Positive indicator 1 (if any)",
    "Positive indicator 2 (if any)"
  ],
  
  "decision_reasoning": "Detailed 3-4 sentence explanation of why you made this decision, referencing specific findings from each agent",
  
  "recommended_action": "Clear, actionable recommendation (e.g., 'Approve transaction', 'Reject and notify user', 'Request additional verification')",
  
  "agent_consensus": "Do the agents agree? Any conflicts in their assessments?",
  
  "confidence_breakdown": {{
    "ml_model_confidence": {fraud_result.get('ml_prediction', {}).get('model_confidence', 0)},
    "behavioral_analysis_confidence": 0.8,
    "biometric_confidence": {deepfake_result.get('verification_summary', {}).get('overall_confidence', 0) / 100 if has_photo else 0},
    "overall_confidence": 0.85
  }}
}}

**IMPORTANT:** 
- Be thorough and reference specific data points
- If agents disagree, explain why
- Consider ALL evidence, not just the ML score
- If biometric verification failed, that's a MAJOR red flag
- Provide clear, actionable guidance

Return ONLY the JSON response, no additional text."""

    return prompt


def assess_risk(transaction_data, fraud_result, evidence_result, deepfake_result):
    """
    Main risk assessment function - synthesizes all agent results
    
    Args:
        transaction_data (dict): Original transaction data
        fraud_result (dict): Results from Transaction Monitor agent
        evidence_result (dict): Results from Evidence Collector agent
        deepfake_result (dict): Results from Deepfake Detector agent (or None)
        
    Returns:
        dict: Comprehensive risk assessment
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 Risk Assessor: Synthesizing All Findings")
    print(f"{'='*60}\n")
    
    try:
        # Build comprehensive prompt
        print("   📝 Building comprehensive analysis prompt...")
        prompt = build_risk_assessment_prompt(
            transaction_data,
            fraud_result,
            evidence_result,
            deepfake_result
        )
        
        # Get detailed assessment from Nova
        print("   🤖 Generating detailed risk assessment with Nova Pro...")
        nova_response = call_nova(prompt, max_tokens=1500, temperature=0.3)
        
        # Parse response
        try:
            assessment = json.loads(nova_response)
            print("   ✅ Risk assessment complete")
        except json.JSONDecodeError:
            print("   ⚠️  Failed to parse Nova response")
            assessment = {
                "final_verdict": "REVIEW",
                "confidence_score": 50,
                "risk_level": "MEDIUM",
                "risk_summary": "Unable to parse comprehensive assessment",
                "key_findings": ["System error in risk assessment"],
                "decision_reasoning": nova_response[:500],
                "recommended_action": "Manual review required"
            }
        
        # Add metadata
        assessment['agent'] = 'risk_assessor'
        assessment['timestamp'] = datetime.now().isoformat()
        assessment['agents_analyzed'] = {
            'fraud_detection': True,
            'evidence_collection': True,
            'biometric_verification': deepfake_result is not None and not deepfake_result.get('error')
        }
        
        print(f"\n   📊 Final Verdict: {assessment.get('final_verdict')}")
        print(f"   📊 Confidence: {assessment.get('confidence_score')}%")
        print(f"   📊 Risk Level: {assessment.get('risk_level')}")
        
        return assessment
        
    except Exception as e:
        print(f"   ❌ Error in risk assessment: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "final_verdict": "REVIEW",
            "confidence_score": 0,
            "risk_level": "UNKNOWN",
            "risk_summary": f"Error during assessment: {str(e)}",
            "key_findings": ["System error"],
            "decision_reasoning": "Unable to complete risk assessment due to system error",
            "recommended_action": "Manual review required",
            "agent": "risk_assessor",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }