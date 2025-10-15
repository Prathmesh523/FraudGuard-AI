"""
Utility Functions
- Suspicious flag calculation
- Input validation
- Feature engineering helpers
"""

from datetime import datetime


def calculate_suspicious_flag(transaction_data, fraud_probability):
    """
    Calculate if transaction is suspicious based on rules + ML prediction
    
    Args:
        transaction_data: Dict with transaction details
        fraud_probability: Float from XGBoost model (0.0 to 1.0)
    
    Returns:
        Tuple (is_suspicious: bool, reasons: list)
    """
    
    reasons = []
    user_id = transaction_data.get('user_id')
    amount = transaction_data.get('transaction_amount', 0)
    
    # Get user baseline
    user_baseline = get_user_baseline(user_id)
    avg_amount = user_baseline.get('avg_transaction_amount', 1000)
    
    # Rule 1: High-value deviation
    if amount > avg_amount * 5 and amount > 5000:
        reasons.append('high_value_deviation')
    
    # Rule 2: ML model high confidence
    if fraud_probability > 0.60:
        reasons.append('ml_high_risk')
    
    # Rule 3: New device + significant amount
    device = transaction_data.get('device_type', 'Desktop')
    known_devices = user_baseline.get('known_devices', [])
    if device not in known_devices and amount > 3000:
        reasons.append('new_device_high_amount')
    
    # Rule 4: Unusual hour + significant amount
    hour = datetime.now().hour
    if (2 <= hour <= 6) and amount > 2000:
        reasons.append('unusual_hour')
    
    # Rule 5: High transaction distance
    distance = transaction_data.get('transaction_distance', 0)
    if distance > 500 and amount > 3000:
        reasons.append('geographic_anomaly')
    
    # Determine if suspicious
    is_suspicious = len(reasons) > 0
    
    return is_suspicious, reasons


def validate_transaction_data(transaction_data):
    """
    Validate transaction input data
    
    Args:
        transaction_data: Dict with transaction details
    
    Returns:
        Tuple (is_valid: bool, error_message: str or None)
    """
    
    required_fields = ['user_id', 'transaction_amount', 'transaction_type', 'merchant_category']
    
    # Check required fields
    for field in required_fields:
        if field not in transaction_data or transaction_data[field] is None:
            return False, f"Missing required field: {field}"
    
    # Validate amount
    amount = transaction_data.get('transaction_amount')
    if not isinstance(amount, (int, float)) or amount <= 0:
        return False, "Invalid transaction amount"
    
    if amount > 1000000:  # $1M limit
        return False, "Transaction amount exceeds maximum limit"
    
    # Validate user_id
    user_id = transaction_data.get('user_id')
    if not isinstance(user_id, str) or len(user_id) == 0:
        return False, "Invalid user_id"
    
    return True, None


def get_user_baseline(user_id):
    """
    Get user behavioral baseline (mock for local dev)
    
    Args:
        user_id: User identifier
    
    Returns:
        Dict with user baseline data
    """
    from services.aws_services import get_user_profile
    
    # Get from DynamoDB (or mock)
    return get_user_profile(user_id)


def extract_features_for_ml(transaction_data):
    """
    Extract and prepare features for ML model
    (Used by fraud_model.py)
    
    Args:
        transaction_data: Dict with transaction details
    
    Returns:
        Dict with engineered features
    """
    # This is handled in fraud_model.py prepare_features()
    # Keeping this function for compatibility
    from services.fraud_model import prepare_features
    return prepare_features(transaction_data)