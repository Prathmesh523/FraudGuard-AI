"""
Agent 3: Evidence Collector
- Gathers user transaction history and behavioral data
- Builds investigation timeline
- Detects suspicious patterns
"""

import sys
sys.path.append('..')

from services.aws_services import (
    get_user_profile,
    get_user_transaction_history,
    call_bedrock_claude
)
from datetime import datetime, timedelta


def collect_evidence(transaction_data):
    """
    Collect evidence for investigation
    
    Args:
        transaction_data: Dict with transaction details
    
    Returns:
        Dict with evidence package
    """
    
    print("   🔍 Collecting evidence...")
    
    user_id = transaction_data.get('user_id')
    
    # =========================================================================
    # Step 1: Get User Profile
    # =========================================================================
    print(f"   📊 Fetching user profile: {user_id}")
    user_profile = get_user_profile(user_id)
    
    # =========================================================================
    # Step 2: Get Transaction History (Last 90 days)
    # =========================================================================
    print("   📜 Retrieving transaction history...")
    transaction_history = get_user_transaction_history(user_id, days=90)
    
    history_count = len(transaction_history)
    recent_24h = [t for t in transaction_history if is_within_24h(t.get('timestamp'))]
    
    print(f"   📈 Total: {history_count} | Last 24h: {len(recent_24h)}")
    
    # =========================================================================
    # Step 3: Build Timeline
    # =========================================================================
    timeline = []
    for txn in transaction_history[-5:]:  # Last 5 transactions
        timeline.append({
            'time': txn.get('timestamp'),
            'event': f"${txn.get('transaction_amount', 0):,.2f} at {txn.get('merchant_category', 'Unknown')}"
        })
    
    timeline.append({
        'time': datetime.now().isoformat(),
        'event': f"CURRENT: ${transaction_data.get('transaction_amount', 0):,.2f} at {transaction_data.get('merchant_category', 'Unknown')}"
    })
    
    # =========================================================================
    # Step 4: Detect Patterns
    # =========================================================================
    patterns = []
    
    current_amount = transaction_data.get('transaction_amount', 0)
    avg_amount = user_profile.get('avg_transaction_amount', 0)
    
    if avg_amount > 0 and current_amount > avg_amount * 5:
        patterns.append(f"Amount {int(current_amount/avg_amount)}x higher than normal")
    
    if len(recent_24h) > 10:
        patterns.append(f"High velocity: {len(recent_24h)} transactions in 24h")
    
    if transaction_data.get('device_type') not in user_profile.get('known_devices', []):
        patterns.append("New device detected")
    
    # =========================================================================
    # Step 5: LLM Summary
    # =========================================================================
    print("   🤖 Generating evidence summary...")
    
    prompt = f"""Summarize this fraud investigation evidence in 2-3 sentences.

User: {user_id}
- Average transaction: ${avg_amount:,.2f}
- Total transactions: {history_count}
- Last 24h: {len(recent_24h)} transactions

Current transaction: ${current_amount:,.2f}

Patterns detected: {', '.join(patterns) if patterns else 'None'}

Focus on key risk indicators."""
    
    llm_summary = call_bedrock_claude(prompt)
    
    return {
        'user_profile': user_profile,
        'transaction_history_count': history_count,
        'recent_24h_count': len(recent_24h),
        'timeline': timeline,
        'detected_patterns': patterns,
        'llm_summary': llm_summary,
        'agent': 'evidence_collector'
    }


def is_within_24h(timestamp):
    """Check if timestamp is within last 24 hours"""
    if not timestamp:
        return False
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timestamp.tzinfo)
        return (now - timestamp) <= timedelta(hours=24)
    except:
        return False