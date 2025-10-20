import json
from datetime import datetime
from tools.deepfake_tools import call_deepfake_api, parse_deepfake_result
from tools.nova_tools import call_nova


def build_deepfake_analysis_prompt(transaction_data, deepfake_result):
    """
    Build prompt for Nova to analyze deepfake detection results
    """
    
    prompt = f"""You are an expert biometric fraud analyst. Analyze this identity verification result and determine if the transaction should proceed.

Transaction Context:
- Transaction ID: {transaction_data.get('transaction_id', 'N/A')}
- User ID: {transaction_data.get('user_id', 'N/A')}
- Transaction Amount: ${transaction_data.get('transaction_amount', 0)}
- Verification Photo: {transaction_data.get('photo_s3_path', 'N/A')}

Biometric Verification Results:
- Overall Result: {deepfake_result['verification_result']}
- Authenticity: {"AUTHENTIC" if deepfake_result['is_authentic'] else "SUSPICIOUS"}
- Overall Confidence: {deepfake_result['overall_confidence']}%

Face Matching:
- Face Match: {"YES" if deepfake_result['face_match']['matched'] else "NO"}
- Similarity Score: {deepfake_result['face_match']['similarity']}%
- Match Confidence: {deepfake_result['face_match']['confidence']}%

Deepfake Detection:
- Is Real Person: {"YES" if deepfake_result['deepfake_detection']['is_real'] else "NO - DEEPFAKE SUSPECTED"}
- Detection Confidence: {deepfake_result['deepfake_detection']['confidence']}%
- Image Quality Score: {deepfake_result['deepfake_detection']['quality_score']}/100

Liveness Check:
- Live Person Detected: {"YES" if deepfake_result['liveness_check']['is_live'] else "NO - PHOTO/VIDEO SUSPECTED"}
- Liveness Score: {deepfake_result['liveness_check']['score']}/100
- Liveness Confidence: {deepfake_result['liveness_check']['confidence']}%

Risk Factors Identified:
{chr(10).join(['- ' + factor for factor in deepfake_result['risk_factors']]) if deepfake_result['risk_factors'] else '- None'}

Image Quality Metrics:
- Brightness: {deepfake_result['image_quality']['brightness']:.2f}
- Sharpness: {deepfake_result['image_quality']['sharpness']:.2f}

Behavioral Indicators:
- Detected Emotion: {deepfake_result['biometrics']['top_emotion']} ({deepfake_result['biometrics']['emotion_confidence']:.1f}%)
- Head Pose: Roll={deepfake_result['biometrics']['pose'].get('roll', 0):.1f}°, Yaw={deepfake_result['biometrics']['pose'].get('yaw', 0):.1f}°, Pitch={deepfake_result['biometrics']['pose'].get('pitch', 0):.1f}°

Based on all verification checks, provide your analysis:

1. **Final Verdict**: APPROVED, REJECTED, or REVIEW
2. **Risk Assessment**: List 2-4 key findings (either security concerns or positive indicators)
3. **Recommended Action**: What should happen with this verification?
4. **Reasoning**: 2-3 sentences explaining your decision, focusing on identity verification confidence

Respond in JSON format:
{{
  "verdict": "APPROVED/REJECTED/REVIEW",
  "risk_assessment": ["finding1", "finding2", "finding3"],
  "recommended_action": "action description",
  "reasoning": "explanation here"
}}

Provide ONLY the JSON response."""

    return prompt


def analyze_deepfake_verification(transaction_data):
    """
    Main deepfake detection agent
    
    Args:
        transaction_data (dict): Must include:
            - user_id
            - photo_s3_path (uploaded verification photo)
            - transaction_id
            - transaction_amount
            - similarity_threshold (optional, default 80)
            
    Returns:
        dict: Complete deepfake analysis with verdict
    """
    
    print(f"\n{'='*60}")
    print(f"🎭 Analyzing Biometric Verification: {transaction_data.get('transaction_id', 'Unknown')}")
    print(f"{'='*60}\n")
    
    try:
        user_id = transaction_data.get('user_id')
        photo_s3_path = transaction_data.get('photo_s3_path')
        similarity_threshold = transaction_data.get('similarity_threshold', 80)
        
        if not user_id or not photo_s3_path:
            raise ValueError("Missing required fields: user_id and photo_s3_path")
        
        # Step 1: Call deepfake detection API
        print("Step 1: Running biometric verification checks...")
        api_response = call_deepfake_api(user_id, photo_s3_path, similarity_threshold)
        
        # Step 2: Parse results
        print("Step 2: Analyzing verification results...")
        deepfake_result = parse_deepfake_result(api_response)
        
        verdict_emoji = "✅" if deepfake_result['is_authentic'] else "⚠️"
        print(f"{verdict_emoji} Verification: {deepfake_result['verification_result']}")
        print(f"   Face Match: {deepfake_result['face_match']['similarity']:.1f}%")
        print(f"   Deepfake Check: {'PASS' if deepfake_result['deepfake_detection']['is_real'] else 'FAIL'}")
        print(f"   Liveness Check: {'PASS' if deepfake_result['liveness_check']['is_live'] else 'FAIL'}")
        
        # Step 3: Get Nova's expert analysis
        print("\nStep 3: Consulting Nova Pro for expert verification analysis...")
        prompt = build_deepfake_analysis_prompt(transaction_data, deepfake_result)
        nova_response = call_nova(prompt, max_tokens=800, temperature=0.2)
        print("✅ Nova analysis complete")
        
        # Step 4: Parse Nova's response
        try:
            nova_analysis = json.loads(nova_response)
        except json.JSONDecodeError:
            nova_analysis = {
                "verdict": "REVIEW",
                "risk_assessment": ["Unable to parse analysis"],
                "recommended_action": "Manual review required",
                "reasoning": nova_response
            }
        
        # Step 5: Combine results
        final_result = {
            "transaction_id": str(transaction_data.get('transaction_id')),
            "user_id": str(user_id),
            "timestamp": datetime.now().isoformat(),
            "verification_summary": {
                "is_authentic": deepfake_result['is_authentic'],
                "overall_confidence": deepfake_result['overall_confidence'],
                "verification_result": deepfake_result['verification_result']
            },
            "detailed_checks": deepfake_result,
            "nova_analysis": nova_analysis,
            "final_verdict": str(nova_analysis.get('verdict', 'REVIEW')),
            "photo_verified": photo_s3_path
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Verification Complete - Verdict: {final_result['final_verdict']}")
        print(f"{'='*60}\n")
        
        return final_result
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        raise