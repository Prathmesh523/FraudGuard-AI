from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv
import random

# Import our modules
from orchestrator import orchestrate_transaction
from services.fraud_model import load_fraud_model, predict_fraud_probability
from services.aws_services import (
    save_pending_transaction,
    get_pending_transaction,
    save_uploaded_photo
)
from utils import calculate_suspicious_flag, validate_transaction_data

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard AI",
    description="Multi-Agent Fraud Detection System with Deepfake Defense",
    version="1.0.0"
)

# Configure CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TransactionRequest(BaseModel):
    user_id: str
    transaction_amount: float
    transaction_type: str
    merchant_category: str
    card_type: str
    device_type: Optional[str] = "Desktop"
    location: Optional[str] = "Unknown"
    authentication_method: Optional[str] = "Password"

class TransactionResponse(BaseModel):
    status: str  # "APPROVED", "BLOCKED", "REVIEW", "VERIFICATION_REQUIRED"
    transaction_id: str
    risk_score: Optional[int] = None
    verification_code: Optional[str] = None
    reasons: Optional[list] = None
    result: Optional[dict] = None

class VerificationRequest(BaseModel):
    transaction_id: str
    verification_code: str

# ============================================================================
# STARTUP EVENT - Load Models Once
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load ML models and initialize connections on startup"""
    print("🚀 Starting FraudGuard AI Backend...")
    
    # Load XGBoost model and encoders
    print("📦 Loading fraud detection model...")
    load_fraud_model()
    print("✅ Model loaded successfully")
    
    # Validate AWS credentials
    print("🔐 Validating AWS credentials...")
    required_env_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "DYNAMODB_TRANSACTIONS_TABLE",
        "S3_BUCKET_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {missing_vars}")
    else:
        print("✅ AWS credentials validated")
    
    print("✅ FraudGuard AI Backend ready!")

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FraudGuard AI",
        "version": "1.0.0"
    }

# ============================================================================
# ENDPOINT 1: Submit Transaction
# ============================================================================

@app.post("/api/transaction/submit", response_model=TransactionResponse)
async def submit_transaction(transaction: TransactionRequest):
    """
    Initial transaction submission with pre-screening
    
    Flow:
    1. Validate input
    2. Run XGBoost pre-screening
    3. Calculate suspicious flag
    4. IF not suspicious -> Call orchestrator, return result
    5. IF suspicious -> Generate code, return verification request
    """
    
    try:
        print(f"\n📥 Received transaction from user: {transaction.user_id}")
        
        # Generate transaction ID
        transaction_id = f"TXN_{random.randint(10000, 99999)}"
        
        # Convert to dict for processing
        transaction_data = transaction.dict()
        transaction_data['transaction_id'] = transaction_id
        
        # Validate transaction data
        is_valid, error_msg = validate_transaction_data(transaction_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # ==========================================
        # PRE-SCREENING: XGBoost + Suspicious Flag
        # ==========================================
        print("🔍 Running pre-screening analysis...")
        
        # Get fraud probability from XGBoost
        fraud_probability = predict_fraud_probability(transaction_data)
        print(f"   ML Fraud Probability: {fraud_probability:.2%}")
        
        # Calculate suspicious flag using rules + ML prediction
        is_suspicious, flag_reasons = calculate_suspicious_flag(
            transaction_data,
            fraud_probability
        )
        
        print(f"   Suspicious Flag: {is_suspicious}")
        if is_suspicious:
            print(f"   Reasons: {flag_reasons}")
        
        # ==========================================
        # DECISION POINT
        # ==========================================
        
        if is_suspicious:
            # ==========================================
            # SUSPICIOUS -> Request Photo Verification
            # ==========================================
            print("⚠️  Transaction flagged as suspicious")
            print("📸 Requesting biometric verification...")
            
            # Generate 6-digit verification code
            verification_code = f"{random.randint(100000, 999999)}"
            
            # Save transaction as PENDING in DynamoDB
            save_pending_transaction(
                transaction_id=transaction_id,
                transaction_data=transaction_data,
                verification_code=verification_code,
                flag_reasons=flag_reasons
            )
            
            print(f"✅ Transaction saved as PENDING: {transaction_id}")
            print(f"🔢 Verification code: {verification_code}")
            
            return TransactionResponse(
                status="VERIFICATION_REQUIRED",
                transaction_id=transaction_id,
                verification_code=verification_code,
                reasons=flag_reasons
            )
        
        else:
            # ==========================================
            # NOT SUSPICIOUS -> Process Immediately
            # ==========================================
            print("✅ Transaction not suspicious - processing normally")
            
            # Call orchestrator (without photo)
            result = orchestrate_transaction(
                transaction_data=transaction_data,
                photo_data=None,
                verification_code=None
            )
            
            print(f"📊 Risk Score: {result['risk_score']}/100")
            print(f"🎯 Final Status: {result['status']}")
            
            return TransactionResponse(
                status=result['status'],
                transaction_id=transaction_id,
                risk_score=result['risk_score'],
                result=result
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ENDPOINT 2: Verify Biometric
# ============================================================================

@app.post("/api/transaction/verify", response_model=TransactionResponse)
async def verify_biometric(
    transaction_id: str,
    verification_code: str,
    photo: UploadFile = File(...)
):
    """
    Process biometric verification and complete analysis
    
    Flow:
    1. Retrieve pending transaction
    2. Upload photo to S3
    3. Call orchestrator with photo
    4. Return final result
    """
    
    try:
        print(f"\n📸 Received verification for transaction: {transaction_id}")
        
        # ==========================================
        # Retrieve Pending Transaction
        # ==========================================
        print("🔍 Retrieving pending transaction...")
        
        transaction_data = get_pending_transaction(transaction_id)
        
        if not transaction_data:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {transaction_id} not found or already processed"
            )
        
        # Validate verification code
        stored_code = transaction_data.get('verification_code')
        if stored_code != verification_code:
            print(f"❌ Invalid verification code")
            raise HTTPException(
                status_code=400,
                detail="Invalid verification code"
            )
        
        print("✅ Transaction retrieved successfully")
        
        # ==========================================
        # Upload Photo to S3
        # ==========================================
        print("📤 Uploading photo to S3...")
        
        # Read photo data
        photo_bytes = await photo.read()
        
        # Upload to S3 and get path
        photo_s3_path = save_uploaded_photo(
            transaction_id=transaction_id,
            photo_data=photo_bytes,
            filename=photo.filename
        )
        
        print(f"✅ Photo uploaded: {photo_s3_path}")
        
        # ==========================================
        # Call Orchestrator with Photo
        # ==========================================
        print("🤖 Invoking agent orchestrator with verification data...")
        
        result = orchestrate_transaction(
            transaction_data=transaction_data,
            photo_data={
                's3_path': photo_s3_path,
                'filename': photo.filename
            },
            verification_code=verification_code
        )
        
        print(f"📊 Risk Score: {result['risk_score']}/100")
        print(f"🎯 Final Status: {result['status']}")
        
        return TransactionResponse(
            status=result['status'],
            transaction_id=transaction_id,
            risk_score=result['risk_score'],
            result=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error verifying biometric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    print(f"❌ Unexpected error: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev only)
    )