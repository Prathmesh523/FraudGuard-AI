from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from orchestrator import orchestrate_fraud_detection_sync
import uvicorn
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

app = FastAPI(
    title="FraudGuard AI Agent System",
    description="Multi-agent fraud detection system with ML, behavioral analysis, and biometric verification",
    version="1.0.0"
)


# Request model
class FraudDetectionRequest(BaseModel):
    # Required fields from Lambda
    transaction_id: str
    user_id: str
    transaction_amount: float
    transaction_type: str
    merchant_category: str
    card_type: str
    
    # Optional fields with defaults
    device_type: Optional[str] = "Desktop"
    location: Optional[str] = "Unknown"
    authentication_method: Optional[str] = "Password"
    
    # Optional photo for biometric verification
    photo_s3_path: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_12345",
                "user_id": "USER_1834",
                "transaction_amount": 2500.00,
                "transaction_type": "ATM Withdrawal",
                "merchant_category": "Electronics",
                "card_type": "Visa",
                "device_type": "Mobile",
                "location": "Tokyo",
                "authentication_method": "Password",
                "photo_s3_path": "s3://bucket/verify/photo.jpg"
            }
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FraudGuard AI Agent System",
        "version": "1.0.0",
        "agents": {
            "agent_1": "Transaction Monitor (ML Fraud Detection)",
            "agent_2": "Evidence Collector (Behavioral Analysis)",
            "agent_3": "Deepfake Detector (Biometric Verification)",
            "agent_4": "Risk Assessor (Decision Synthesis)"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agents_available": ["transaction_monitor", "evidence_collector", "deepfake_detector", "risk_assessor"]
    }


@app.post("/fraud-detection")
def detect_fraud(request: FraudDetectionRequest):
    """
    Main fraud detection endpoint
    
    Receives transaction data from Lambda and returns comprehensive fraud analysis
    from all 4 AI agents running in parallel.
    
    Args:
        request: FraudDetectionRequest with transaction details and optional photo
        
    Returns:
        Complete fraud detection result with verdict, risk assessment, and agent analyses
    """
    
    try:
        # Convert request to dict for orchestrator
        transaction_data = {
            'transaction_id': request.transaction_id,
            'user_id': request.user_id,
            'transaction_amount': request.transaction_amount,
            'transaction_type': request.transaction_type,
            'merchant_category': request.merchant_category,
            'card_type': request.card_type,
            'device_type': request.device_type,
            'location': request.location,
            'authentication_method': request.authentication_method
        }
        
        # Run orchestrator (all 4 agents)
        result = orchestrate_fraud_detection_sync(
            transaction_data=transaction_data,
            photo_s3_path=request.photo_s3_path
        )
        
        return convert_to_serializable(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Fraud detection failed",
                "message": str(e),
                "transaction_id": request.transaction_id
            }
        )


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)