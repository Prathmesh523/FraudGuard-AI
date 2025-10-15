"""
Agent 4: Risk Assessor
- Synthesizes findings from all agents
- Calculates composite risk score
- Provides final recommendation
"""

import sys
sys.path.append('..')

from services.aws_services import call_bedrock_claude


def assess_risk(monitor_result, deepfake_result, evidence):
    """
    Synthesize all agent findings into final risk assessment
    
    Args:
        monitor_result: Output from Agent 1 (Transaction Monitor)
        deepfake_result: Output from Agent 2 (Deepfake Detector) or None
        evidence: Output from Agent 3 (Evidence Collector)
    
    Returns:
        Dict with final risk assessment
    """
    
    print("   ⚖️  Synthesizing risk assessment...")
    
    # =========================================================================
    # Step 1: Calculate Composite Risk Score
    # =========================================================================
    
    # Base score from ML model
    base_risk = monitor_result['risk_score']
    
    # Adjustments based on other signals
    risk_adjustments = []
    
    # Deepfake detection (if checked)
    if deepfake_result:
        if deepfake_result['is_deepfake']:
            risk_adjustments.append(('Deepfake detected', +40))
        elif deepfake_result['face_match_score'] < 0.80:
            risk_adjustments.append(('Low face match', +20))
        elif not deepfake_result['code_validated']:
            risk_adjustments.append(('Code validation failed', +15))
    
    # Evidence patterns
    patterns = evidence.get('detected_patterns', [])
    if any('higher than normal' in p for p in patterns):
        risk_adjustments.append(('Amount anomaly', +15))
    if any('High velocity' in p for p in patterns):
        risk_adjustments.append(('Velocity spike', +10))
    if any('New device' in p for p in patterns):
        risk_adjustments.append(('New device', +10))
    
    # Calculate final score (capped at 100)
    adjustment_total = sum(adj[1] for adj in risk_adjustments)
    composite_risk_score = min(base_risk + adjustment_total, 100)
    
    print(f"   📊 Base Risk: {base_risk}")
    print(f"   📈 Adjustments: +{adjustment_total}")
    print(f"   🎯 Composite Risk: {composite_risk_score}/100")
    
    # =========================================================================
    # Step 2: Determine Status
    # =========================================================================
    if composite_risk_score >= 85:
        status = "BLOCKED"
        confidence = "very_high"
    elif composite_risk_score >= 70:
        status = "REVIEW"
        confidence = "high"
    elif composite_risk_score >= 50:
        status = "REVIEW"
        confidence = "medium"
    else:
        status = "APPROVED"
        confidence = "low_risk"
    
    # =========================================================================
    # Step 3: LLM Reasoning Chain
    # =========================================================================
    print("   🤖 Generating reasoning via Bedrock Claude...")
    
    prompt = f"""You are a senior fraud analyst making a final decision. Synthesize these findings.

Transaction Monitor (Agent 1):
- ML Risk Score: {base_risk}/100
- Explanation: {monitor_result.get('llm_explanation', 'N/A')}

{"Deepfake Detector (Agent 2):" if deepfake_result else ""}
{f"- Deepfake: {deepfake_result['is_deepfake']}" if deepfake_result else ""}
{f"- Face Match: {deepfake_result['face_match_score']:.0%}" if deepfake_result else ""}
{f"- Explanation: {deepfake_result.get('llm_explanation', 'N/A')}" if deepfake_result else ""}

Evidence Collector (Agent 3):
- Patterns: {', '.join(patterns) if patterns else 'None'}
- Summary: {evidence.get('llm_summary', 'N/A')}

Risk Adjustments Applied:
{chr(10).join(f'- {reason}: +{value}' for reason, value in risk_adjustments) if risk_adjustments else '- None'}

Composite Risk Score: {composite_risk_score}/100
Recommended Status: {status}

Provide a 3-4 sentence executive summary explaining why this transaction should be {status}. Connect the findings logically."""
    
    llm_reasoning = call_bedrock_claude(prompt)
    
    print(f"   ✅ Reasoning: {llm_reasoning[:100]}...")
    
    return {
        'risk_score': composite_risk_score,
        'base_risk': base_risk,
        'adjustments': risk_adjustments,
        'status': status,
        'confidence': confidence,
        'llm_reasoning': llm_reasoning,
        'agent': 'risk_assessor'
    }