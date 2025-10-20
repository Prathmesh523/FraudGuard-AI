import json
from datetime import datetime, timedelta
from tools.dynamodb_tools import (
    get_user_profile,
    get_user_transaction_history,
    get_recent_transactions_count
)
from tools.nova_tools import call_nova


def is_within_24h(timestamp):
    """Check if timestamp is within last 24 hours"""
    if not timestamp:
        return False
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        return (now - timestamp) <= timedelta(hours=24)
    except:
        return False


def build_timeline(transaction_history, current_transaction):
    """
    Build chronological timeline of events
    
    Args:
        transaction_history (list): Past transactions
        current_transaction (dict): Current transaction being analyzed
        
    Returns:
        list: Timeline events
    """
    
    timeline = []
    
    # Add last 5 transactions
    for txn in transaction_history[-5:]:
        timeline.append({
            'time': txn.get('Timestamp'),
            'event': f"${float(txn.get('Transaction_Amount', 0)):,.2f} at {txn.get('Merchant_Category', 'Unknown')} ({txn.get('Location', 'Unknown')})",
            'type': txn.get('Transaction_Type', 'Unknown'),
            'fraud_flag': int(txn.get('Fraud_Label', 0))
        })
    
    # Add current transaction
    timeline.append({
        'time': datetime.now().isoformat(),
        'event': f"CURRENT: ${float(current_transaction.get('transaction_amount', 0)):,.2f} at {current_transaction.get('merchant_category', 'Unknown')} ({current_transaction.get('location', 'Unknown')})",
        'type': current_transaction.get('transaction_type', 'Unknown'),
        'fraud_flag': 0
    })
    
    return timeline


def detect_patterns(transaction_data, user_profile, transaction_history):
    """
    Detect suspicious patterns in user behavior
    
    Args:
        transaction_data (dict): Current transaction
        user_profile (dict): User's historical profile
        transaction_history (list): Recent transactions
        
    Returns:
        list: Detected patterns/anomalies
    """
    
    patterns = []
    
    current_amount = transaction_data.get('transaction_amount', 0)
    avg_amount = user_profile.get('avg_transaction_amount', 0)
    
    # Pattern 1: Amount anomaly
    if avg_amount > 0 and current_amount > avg_amount * 5:
        multiplier = int(current_amount / avg_amount)
        patterns.append(f"⚠️ Amount {multiplier}x higher than user average (${avg_amount:.2f})")
    
    # Pattern 2: High velocity
    recent_24h = [t for t in transaction_history if is_within_24h(t.get('Timestamp'))]
    if len(recent_24h) > 10:
        patterns.append(f"⚠️ High velocity: {len(recent_24h)} transactions in last 24 hours")
    
    # Pattern 3: New device
    current_device = transaction_data.get('device_type')
    known_devices = user_profile.get('known_devices', [])
    if current_device and current_device not in known_devices:
        patterns.append(f"⚠️ New device detected: {current_device}")
    
    # Pattern 4: New location
    current_location = transaction_data.get('location')
    known_locations = user_profile.get('known_locations', [])
    if current_location and current_location not in known_locations:
        patterns.append(f"⚠️ New location: {current_location}")
    
    # Pattern 5: Unusual merchant
    current_merchant = transaction_data.get('merchant_category')
    known_merchants = user_profile.get('known_merchants', [])
    if current_merchant and current_merchant not in known_merchants:
        patterns.append(f"ℹ️ First transaction at {current_merchant}")
    
    # Pattern 6: Fraud history
    if user_profile.get('fraud_history', 0) > 0:
        patterns.append(f"🚨 User has {user_profile['fraud_history']} previous fraud flags")
    
    # Pattern 7: Rapid successive transactions
    if len(recent_24h) >= 3:
        # Check if transactions are within 1 hour of each other
        recent_times = [datetime.fromisoformat(t.get('Timestamp', '').replace('Z', '+00:00')) for t in recent_24h[:3]]
        if len(recent_times) >= 2:
            time_diff = (recent_times[0] - recent_times[-1]).total_seconds() / 3600
            if time_diff < 1:
                patterns.append(f"⚠️ Multiple transactions within 1 hour")
    
    return patterns


def collect_evidence(transaction_data):
    """
    Main evidence collection function
    
    Args:
        transaction_data (dict): Current transaction details
        
    Returns:
        dict: Evidence package
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 Collecting Evidence: {transaction_data.get('transaction_id', 'Unknown')}")
    print(f"{'='*60}\n")
    
    try:
        user_id = transaction_data.get('user_id')
        
        if not user_id:
            raise ValueError("user_id is required")
        
        # Step 1: Get User Profile
        print("Step 1: Fetching user profile...")
        user_profile = get_user_profile(user_id)
        
        # Step 2: Get Transaction History
        print("\nStep 2: Retrieving transaction history...")
        # Get all history (no date filter since CSV has old data)
        transaction_history = get_user_transaction_history(user_id, days=None, limit=100)
        
        history_count = len(transaction_history)
        recent_24h_count = get_recent_transactions_count(user_id, hours=24)
        
        print(f"   📈 Total: {history_count} | Last 24h: {recent_24h_count}")
        
        # Step 3: Build Timeline
        print("\nStep 3: Building timeline...")
        timeline = build_timeline(transaction_history, transaction_data)
        
        # Step 4: Detect Patterns
        print("\nStep 4: Detecting suspicious patterns...")
        patterns = detect_patterns(transaction_data, user_profile, transaction_history)
        
        if patterns:
            print(f"   🔴 Found {len(patterns)} patterns:")
            for pattern in patterns:
                print(f"      {pattern}")
        else:
            print("   ✅ No suspicious patterns detected")
        
        # Step 5: Generate LLM Summary
        print("\nStep 5: Generating evidence summary with Nova...")

        # Safe access to user profile with defaults
        avg_amount = user_profile.get('avg_transaction_amount', 0)
        min_amount = user_profile.get('min_transaction', 0)
        max_amount = user_profile.get('max_transaction', 0)
        total_txns = user_profile.get('total_transactions', 0)
        known_devices = user_profile.get('known_devices', [])
        fraud_history = user_profile.get('fraud_history', 0)

        prompt = f"""You are a financial transaction analyst. Provide a brief 2-3 sentence analysis of this user's transaction profile.

        **User Activity Summary:**
        - User ID: {user_id}
        - Historical Transactions: {total_txns}
        - Typical Transaction: ${avg_amount:.2f}
        - Range: ${min_amount:.2f} - ${max_amount:.2f}
        - Devices Used: {', '.join(known_devices) if known_devices else 'Unknown'}
        - Previous Issues: {fraud_history} flags

        **Current Transaction Under Review:**
        - Amount: ${transaction_data.get('transaction_amount', 0):.2f}
        - Category: {transaction_data.get('merchant_category', 'Unknown')}
        - Location: {transaction_data.get('location', 'Unknown')}
        - Device: {transaction_data.get('device_type', 'Unknown')}

        **Activity Metrics:**
        - Recent (24h): {recent_24h_count} transactions
        - Historical (90d): {history_count} transactions

        **Notable Observations:**
        {chr(10).join(patterns) if patterns else 'Standard transaction pattern'}

        Provide a professional summary highlighting key observations about this transaction relative to the user's normal behavior."""

        try:
            llm_summary = call_nova(prompt, max_tokens=300, temperature=0.3)
            print("   ✅ Summary generated")
        except Exception as e:
            print(f"   ⚠️  Nova summary failed: {e}")
            llm_summary = "Unable to generate summary due to system error."
        
        # Build evidence package
        evidence = {
            'user_profile': {
                'user_id': user_id,
                'total_transactions': total_txns,
                'avg_transaction_amount': round(avg_amount, 2),
                'transaction_range': {
                    'min': round(min_amount, 2),
                    'max': round(max_amount, 2)
                },
                'known_devices': known_devices,
                'known_locations': user_profile.get('known_locations', [])[:5],
                'fraud_history': fraud_history
            },
            'transaction_history': {
                'total_count': history_count,
                'recent_24h_count': recent_24h_count,
                'recent_transactions': [
                    {
                        'amount': float(t.get('Transaction_Amount', 0)),
                        'merchant': t.get('Merchant_Category'),
                        'time': t.get('Timestamp'),
                        'fraud_flag': int(t.get('Fraud_Label', 0))
                    }
                    for t in transaction_history[:5]
                ]
            },
            'timeline': timeline,
            'detected_patterns': patterns,
            'llm_summary': llm_summary,
            'agent': 'evidence_collector',
            'timestamp': datetime.now().isoformat()
        }
        print(f"\n{'='*60}")
        print(f"✅ Evidence Collection Complete")
        print(f"{'='*60}\n")
        
        return evidence
        
    except Exception as e:
        print(f"\n❌ Error collecting evidence: {e}")
        import traceback
        traceback.print_exc()
        raise