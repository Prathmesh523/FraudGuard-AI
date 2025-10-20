import boto3
import json
from config import AWS_REGION

def call_deepfake_api(user_id, uploaded_image_s3_uri, similarity_threshold=80):
    """
    Call SageMaker deepfake detection endpoint via API Gateway
    
    Args:
        user_id (str): User identifier
        uploaded_image_s3_uri (str): S3 URI of uploaded verification photo
        similarity_threshold (int): Minimum similarity score (0-100)
        
    Returns:
        dict: Verification results from API
    """
    
    # You'll need to set these in config
    from config import DEEPFAKE_API_ENDPOINT
    
    import requests
    
    payload = {
        "user_id": user_id,
        "uploaded_image_s3_uri": uploaded_image_s3_uri,
        "similarity_threshold": similarity_threshold
    }
    
    try:
        print(f"Calling deepfake detection API for user {user_id}...")
        
        response = requests.post(
            DEEPFAKE_API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Deepfake API responded: {result.get('verification_result')}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling deepfake API: {e}")
        raise


def parse_deepfake_result(api_response):
    """
    Parse and structure the deepfake API response
    
    Args:
        api_response (dict): Raw API response
        
    Returns:
        dict: Structured verification analysis
    """
    
    verification_result = api_response.get('verification_result', 'UNKNOWN')
    
    # Extract key metrics
    face_comparison = api_response.get('face_comparison', {})
    quality_check = api_response.get('quality_check', {})
    liveness_check = api_response.get('liveness_check', {})
    
    # Determine overall authenticity
    is_authentic = (
        verification_result == 'VERIFIED' and
        face_comparison.get('match', False) and
        quality_check.get('is_real', False) and
        liveness_check.get('is_live', False)
    )
    
    # Calculate confidence score (average of all checks)
    confidences = [
        face_comparison.get('confidence', 0),
        quality_check.get('confidence', 0),
        liveness_check.get('confidence', 0)
    ]
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Identify risk factors
    risk_factors = []
    
    if not face_comparison.get('match'):
        risk_factors.append(f"Face mismatch (similarity: {face_comparison.get('similarity', 0)}%)")
    
    if not quality_check.get('is_real'):
        risk_factors.append("Potential deepfake detected")
    
    if not liveness_check.get('is_live'):
        risk_factors.append("Liveness check failed")
    
    if quality_check.get('quality_score', 100) < 70:
        risk_factors.append(f"Low image quality ({quality_check.get('quality_score')})")
    
    # Check for suspicious patterns
    checks = liveness_check.get('checks', {})
    if not checks.get('eyes_open'):
        risk_factors.append("Eyes not visible")
    if checks.get('no_sunglasses') == False:
        risk_factors.append("Sunglasses detected")
    
    # Get emotions (can indicate stress/deception)
    emotions = quality_check.get('emotions', [])
    top_emotion = emotions[0] if emotions else {}
    
    return {
        "is_authentic": is_authentic,
        "verification_result": verification_result,
        "overall_confidence": round(overall_confidence, 2),
        "face_match": {
            "matched": face_comparison.get('match', False),
            "similarity": face_comparison.get('similarity', 0),
            "confidence": face_comparison.get('confidence', 0)
        },
        "deepfake_detection": {
            "is_real": quality_check.get('is_real', False),
            "confidence": quality_check.get('confidence', 0),
            "quality_score": quality_check.get('quality_score', 0)
        },
        "liveness_check": {
            "is_live": liveness_check.get('is_live', False),
            "score": liveness_check.get('liveness_score', 0),
            "confidence": liveness_check.get('confidence', 0),
            "checks_passed": checks
        },
        "image_quality": {
            "brightness": quality_check.get('quality_metrics', {}).get('brightness', 0),
            "sharpness": quality_check.get('quality_metrics', {}).get('sharpness', 0)
        },
        "biometrics": {
            "pose": quality_check.get('pose', {}),
            "top_emotion": top_emotion.get('type', 'UNKNOWN'),
            "emotion_confidence": top_emotion.get('confidence', 0)
        },
        "risk_factors": risk_factors,
        "raw_reason": api_response.get('reason', '')
    }