import asyncio
import json
from datetime import datetime
from agents.transaction_monitor import analyze_transaction as analyze_fraud
from agents.evidence_collector import collect_evidence
from agents.deepfake_detector import analyze_deepfake_verification
from agents.risk_assessor import assess_risk


async def run_transaction_monitor(transaction_data):
    """Run fraud detection analysis"""
    try:
        print("   🔍 Starting Transaction Monitor...")
        result = analyze_fraud(transaction_data)
        print("   ✅ Transaction Monitor complete")
        return result
    except Exception as e:
        print(f"   ❌ Transaction Monitor failed: {e}")
        return {"error": str(e), "agent": "transaction_monitor"}


async def run_evidence_collector(transaction_data):
    """Run evidence collection"""
    try:
        print("   📊 Starting Evidence Collector...")
        result = collect_evidence(transaction_data)
        print("   ✅ Evidence Collector complete")
        return result
    except Exception as e:
        print(f"   ❌ Evidence Collector failed: {e}")
        return {"error": str(e), "agent": "evidence_collector"}


async def run_deepfake_detector(transaction_data, photo_s3_path):
    """Run deepfake detection (optional - only if photo provided)"""
    if not photo_s3_path:
        print("   ⏭️  No photo provided, skipping Deepfake Detector")
        return None
    
    try:
        print("   🎭 Starting Deepfake Detector...")
        result = analyze_deepfake_verification({
            **transaction_data,
            'photo_s3_path': photo_s3_path
        })
        print("   ✅ Deepfake Detector complete")
        return result
    except Exception as e:
        print(f"   ❌ Deepfake Detector failed: {e}")
        return {"error": str(e), "agent": "deepfake_detector"}


async def orchestrate_fraud_detection(transaction_data, photo_s3_path=None):
    """
    Main orchestrator - runs all agents and synthesizes results
    
    Args:
        transaction_data (dict): Transaction details from Lambda
        photo_s3_path (str): Optional S3 path to verification photo
        
    Returns:
        dict: Complete fraud detection result
    """
    
    print("\n" + "="*80)
    print("🚀 ORCHESTRATOR: Starting Fraud Detection Pipeline")
    print("="*80)
    print(f"Transaction ID: {transaction_data.get('transaction_id')}")
    print(f"User ID: {transaction_data.get('user_id')}")
    print(f"Amount: ${transaction_data.get('transaction_amount', 0):,.2f}")
    print(f"Photo Verification: {'YES' if photo_s3_path else 'NO'}")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    # =========================================================================
    # PHASE 1: Run Agents 1, 2, 3 in parallel
    # =========================================================================
    print("📡 PHASE 1: Running Specialized Agents (Parallel)\n")
    
    tasks = [
        run_transaction_monitor(transaction_data),
        run_evidence_collector(transaction_data),
        run_deepfake_detector(transaction_data, photo_s3_path)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    fraud_result = results[0]
    evidence_result = results[1]
    deepfake_result = results[2]
    
    print("\n" + "="*80)
    print("✅ PHASE 1 Complete - All Agents Finished")
    print("="*80 + "\n")
    
    # =========================================================================
    # PHASE 2: Agent 4 - Risk Assessor (synthesizes all results)
    # =========================================================================
    print("📡 PHASE 2: Risk Assessment & Final Decision\n")
    
    risk_assessment = assess_risk(
        transaction_data,
        fraud_result,
        evidence_result,
        deepfake_result
    )
    
    print("\n" + "="*80)
    print("✅ PHASE 2 Complete - Risk Assessment Done")
    print("="*80 + "\n")
    
    # =========================================================================
    # PHASE 3: Build complete response
    # =========================================================================
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    complete_result = {
        "transaction_id": transaction_data.get('transaction_id'),
        "user_id": transaction_data.get('user_id'),
        "timestamp": end_time.isoformat(),
        "processing_time_seconds": round(processing_time, 2),
        
        # Final comprehensive assessment from Agent 4
        "risk_assessment": risk_assessment,
        
        # Individual agent results
        "agent_results": {
            "agent_1_fraud_detection": fraud_result,
            "agent_2_evidence_collection": evidence_result,
            "agent_3_biometric_verification": deepfake_result,
            "agent_4_risk_assessment": risk_assessment
        },
        
        # Quick summary
        "summary": {
            "final_verdict": risk_assessment.get('final_verdict'),
            "confidence": risk_assessment.get('confidence_score'),
            "risk_level": risk_assessment.get('risk_level'),
            "ml_fraud_score": fraud_result.get('final_score', 0),
            "suspicious_patterns": len(evidence_result.get('detected_patterns', [])),
            "photo_verified": deepfake_result is not None and not deepfake_result.get('error'),
            "recommended_action": risk_assessment.get('recommended_action')
        }
    }
    
    print("\n" + "="*80)
    print(f"🎯 FINAL VERDICT: {risk_assessment.get('final_verdict')}")
    print(f"   Risk Level: {risk_assessment.get('risk_level')}")
    print(f"   Confidence: {risk_assessment.get('confidence_score')}%")
    print(f"   Action: {risk_assessment.get('recommended_action')}")
    print(f"   Processing Time: {processing_time:.2f}s")
    print("="*80 + "\n")
    
    return complete_result


# Sync wrapper for FastAPI
def orchestrate_fraud_detection_sync(transaction_data, photo_s3_path=None):
    """Synchronous wrapper for orchestrator"""
    return asyncio.run(orchestrate_fraud_detection(transaction_data, photo_s3_path))