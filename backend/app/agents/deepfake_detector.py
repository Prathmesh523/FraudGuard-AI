"""
Agent 2: Deepfake Detector
- Validates biometric authentication
- Detects AI-generated deepfakes
- Verifies face match and code validation
"""

import sys
sys.path.append('..')

from services.aws_services import invoke_sagemaker_endpoint, get_reference_photo, call_bedrock_claude


def analyze_deepfake(transaction_data, photo_s3_path, expected_code):
    """
    Analyze uploaded photo for deepfake and verify authentication
    
    Args:
        transaction_data: Dict with transaction details
        photo_s3_path: S3 path to uploaded photo
        expected_code: 6-digit verification code
    
    Returns:
        Dict with deepfake detection results
    """
    
    print("   🎭 Analyzing biometric verification...")
    
    user_id = transaction_data.get('user_id')
    
    # =========================================================================
    # Step 1: Get Reference Photo
    # =========================================================================
    print(f"   📷 Fetching reference photo for user: {user_id}")
    reference_photo_path = get_reference_photo(user_id)
    
    # =========================================================================
    # Step 2: Call SageMaker Endpoint (MesoNet + DeepFace)
    # =========================================================================
    print("   🔬 Running deepfake detection model...")
    
    sagemaker_input = {
        'uploaded_photo_s3': photo_s3_path,
        'reference_photo_s3': reference_photo_path,
        'expected_code': expected_code
    }
    
    detection_result = invoke_sagemaker_endpoint(sagemaker_input)
    
    is_deepfake = detection_result.get('is_deepfake', False)
    deepfake_confidence = detection_result.get('deepfake_confidence', 0.0)
    face_match_score = detection_result.get('face_match_score', 0.0)
    code_validated = detection_result.get('code_validated', False)
    
    print(f"   🎯 Deepfake Detected: {is_deepfake} (confidence: {deepfake_confidence:.2%})")
    print(f"   👤 Face Match Score: {face_match_score:.2%}")
    print(f"   🔢 Code Validated: {code_validated}")
    
    # =========================================================================
    # Step 3: Overall Verification Status
    # =========================================================================
    verification_passed = (
        not is_deepfake and 
        face_match_score >= 0.80 and 
        code_validated
    )
    
    # =========================================================================
    # Step 4: LLM Explanation
    # =========================================================================
    print("   🤖 Generating explanation via Bedrock Claude...")
    
    prompt = f"""You are a biometric security expert. Analyze this authentication attempt.

Results:
- Deepfake Detected: {is_deepfake} (confidence: {deepfake_confidence:.2%})
- Face Match Score: {face_match_score:.2%} (threshold: 80%)
- Code Validated: {code_validated}
- Overall Verification: {'PASSED' if verification_passed else 'FAILED'}

Provide a 2-3 sentence explanation of why this verification {'passed' if verification_passed else 'failed'}. Focus on security implications.
"""
    
    llm_explanation = call_bedrock_claude(prompt)
    
    print(f"   ✅ Explanation: {llm_explanation[:100]}...")
    
    return {
        'is_deepfake': is_deepfake,
        'deepfake_confidence': deepfake_confidence,
        'face_match_score': face_match_score,
        'code_validated': code_validated,
        'verification_passed': verification_passed,
        'llm_explanation': llm_explanation,
        'agent': 'deepfake_detector'
    }