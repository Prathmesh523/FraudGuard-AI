"""
Agent Orchestrator - Coordinates all 5 agents in sequential workflow
"""

from agents.transaction_monitor import analyze_transaction
from agents.deepfake_detector import analyze_deepfake
from agents.evidence_collector import collect_evidence
from agents.risk_assessor import assess_risk
from agents.escalation_handler import handle_escalation


def orchestrate_transaction(transaction_data, photo_data=None, verification_code=None):
    """
    Orchestrate multi-agent workflow for fraud detection
    
    Args:
        transaction_data: Dict with transaction details
        photo_data: Dict with S3 path and filename (None if not suspicious)
        verification_code: Expected 6-digit code (None if not suspicious)
    
    Returns:
        Dict with final analysis result
    """
    
    print("\n🤖 ORCHESTRATOR: Starting multi-agent analysis...")
    
    # =========================================================================
    # AGENT 1: Transaction Monitor (Always Runs)
    # =========================================================================
    print("\n📊 Agent 1: Transaction Monitor")
    monitor_result = analyze_transaction(transaction_data)
    
    # =========================================================================
    # AGENT 2: Deepfake Detector (Only if photo provided)
    # =========================================================================
    deepfake_result = None
    if photo_data and verification_code:
        print("\n🎭 Agent 2: Deepfake Detector")
        deepfake_result = analyze_deepfake(
            transaction_data=transaction_data,
            photo_s3_path=photo_data['s3_path'],
            expected_code=verification_code
        )
        
        # If deepfake detected, block immediately
        if deepfake_result['is_deepfake'] or not deepfake_result['verification_passed']:
            print("❌ DEEPFAKE DETECTED - Immediate block")
            return {
                'status': 'BLOCKED',
                'risk_score': 100,
                'reason': 'Deepfake authentication detected',
                'deepfake_result': deepfake_result
            }
    
    # =========================================================================
    # AGENT 3: Evidence Collector
    # =========================================================================
    print("\n🔍 Agent 3: Evidence Collector")
    evidence = collect_evidence(transaction_data)
    
    # =========================================================================
    # AGENT 4: Risk Assessor
    # =========================================================================
    print("\n⚖️  Agent 4: Risk Assessor")
    risk_assessment = assess_risk(
        monitor_result=monitor_result,
        deepfake_result=deepfake_result,
        evidence=evidence
    )
    
    # =========================================================================
    # AGENT 5: Escalation Handler (If high risk)
    # =========================================================================
    if risk_assessment['risk_score'] >= 70:
        print("\n🚨 Agent 5: Escalation Handler")
        escalation_result = handle_escalation(
            transaction_data=transaction_data,
            risk_assessment=risk_assessment,
            evidence=evidence
        )
        risk_assessment['case_id'] = escalation_result.get('case_id')
        risk_assessment['alerts_sent'] = escalation_result.get('alerts_sent')
    
    # =========================================================================
    # Final Result
    # =========================================================================
    print(f"\n✅ ORCHESTRATOR: Analysis complete")
    print(f"   Final Risk Score: {risk_assessment['risk_score']}/100")
    print(f"   Status: {risk_assessment['status']}")
    
    return {
        'status': risk_assessment['status'],
        'risk_score': risk_assessment['risk_score'],
        'monitor_result': monitor_result,
        'deepfake_result': deepfake_result,
        'evidence': evidence,
        'risk_assessment': risk_assessment
    }