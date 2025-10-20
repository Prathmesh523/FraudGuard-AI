"""
Agent 5: Escalation Handler
- Creates fraud cases for high-risk transactions
- Sends alerts via SNS
- Stores evidence packages
"""

import sys
sys.path.append('..')

from services.aws_services import (
    create_fraud_case,
    send_sns_alert,
    save_evidence_package,
    call_bedrock_claude
)
from datetime import datetime
import random


def handle_escalation(transaction_data, risk_assessment, evidence):
    """
    Handle escalation for high-risk transactions
    
    Args:
        transaction_data: Dict with transaction details
        risk_assessment: Output from Agent 4 (Risk Assessor)
        evidence: Output from Agent 3 (Evidence Collector)
    
    Returns:
        Dict with escalation details
    """
    
    print("   🚨 Escalating high-risk transaction...")
    
    transaction_id = transaction_data.get('transaction_id')
    risk_score = risk_assessment['risk_score']
    
    # =========================================================================
    # Step 1: Generate Case ID
    # =========================================================================
    case_id = f"FA-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    print(f"   📋 Case ID: {case_id}")
    
    # =========================================================================
    # Step 2: Generate Fraud Case Report via LLM
    # =========================================================================
    print("   📝 Generating fraud case report...")
    
    prompt = f"""Generate a formal fraud case report for investigators.

CASE: {case_id}
TRANSACTION: {transaction_id}
RISK SCORE: {risk_score}/100 - {risk_assessment['status']}

Transaction Details:
- Amount: ${transaction_data.get('transaction_amount', 0):,.2f}
- Type: {transaction_data.get('transaction_type', 'Unknown')}
- Merchant: {transaction_data.get('merchant_category', 'Unknown')}
- User: {transaction_data.get('user_id', 'Unknown')}
- Device: {transaction_data.get('device_type', 'Unknown')}
- Location: {transaction_data.get('location', 'Unknown')}

Risk Assessment:
{risk_assessment.get('llm_reasoning', 'N/A')}

Evidence Summary:
{evidence.get('llm_summary', 'N/A')}

Detected Patterns:
{chr(10).join(f'- {p}' for p in evidence.get('detected_patterns', [])) if evidence.get('detected_patterns') else '- None'}

Format as a professional fraud alert with:
1. Executive summary (2-3 sentences)
2. Key risk indicators
3. Recommended actions"""
    
    fraud_report = call_bedrock_claude(prompt)
    
    # =========================================================================
    # Step 3: Save Evidence Package to S3
    # =========================================================================
    print("   💾 Saving evidence package...")
    
    evidence_package = {
        'case_id': case_id,
        'transaction_id': transaction_id,
        'timestamp': datetime.now().isoformat(),
        'transaction_data': transaction_data,
        'risk_assessment': risk_assessment,
        'evidence': evidence,
        'fraud_report': fraud_report
    }
    
    evidence_s3_path = save_evidence_package(case_id, evidence_package)
    print(f"   📦 Evidence saved: {evidence_s3_path}")
    
    # =========================================================================
    # Step 4: Create Fraud Case in DynamoDB
    # =========================================================================
    print("   💾 Creating fraud case record...")
    
    fraud_case = {
        'case_id': case_id,
        'transaction_id': transaction_id,
        'user_id': transaction_data.get('user_id'),
        'risk_score': risk_score,
        'status': risk_assessment['status'],
        'amount': transaction_data.get('transaction_amount', 0),
        'fraud_report': fraud_report,
        'evidence_s3_path': evidence_s3_path,
        'created_at': datetime.now().isoformat()
    }
    
    create_fraud_case(fraud_case)
    
    # =========================================================================
    # Step 5: Send SNS Alert
    # =========================================================================
    print("   📧 Sending fraud alert via SNS...")
    
    alert_message = f"""🚨 FRAUD ALERT: {case_id}

Risk Score: {risk_score}/100 - {risk_assessment['status']}
Transaction: {transaction_id}
Amount: ${transaction_data.get('transaction_amount', 0):,.2f}

{fraud_report[:500]}...

Full report: {evidence_s3_path}
"""
    
    sns_result = send_sns_alert(
        subject=f"Fraud Alert: {case_id}",
        message=alert_message
    )
    
    print(f"   ✅ Alert sent: {sns_result.get('MessageId', 'N/A')}")
    
    return {
        'case_id': case_id,
        'fraud_report': fraud_report,
        'evidence_s3_path': evidence_s3_path,
        'alerts_sent': [sns_result.get('MessageId')],
        'agent': 'escalation_handler'
    }