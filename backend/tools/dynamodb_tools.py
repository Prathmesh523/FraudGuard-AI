import boto3
from datetime import datetime, timedelta
from decimal import Decimal
from config import DYNAMODB_TABLE_NAME, AWS_REGION

# Cache table resource
_table = None

def get_dynamodb_table():
    """Get DynamoDB table resource (cached)"""
    global _table
    if _table is None:
        dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        _table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    return _table


def get_user_profile(user_id):
    """
    Get aggregated user profile from transaction history
    OPTIMIZED: Uses query with GSI
    """
    
    print(f"   📊 Fetching profile for: {user_id}")
    
    table = get_dynamodb_table()
    
    try:
        # Use GSI query instead of scan - MUCH FASTER!
        response = table.query(
            IndexName='UserIdIndex',
            KeyConditionExpression='User_ID = :uid',
            ExpressionAttributeValues={':uid': user_id}
        )
        
        transactions = response.get('Items', [])
        
        # Handle pagination (still needed but much faster)
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='UserIdIndex',
                KeyConditionExpression='User_ID = :uid',
                ExpressionAttributeValues={':uid': user_id},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            transactions.extend(response.get('Items', []))
        
        if not transactions:
            print(f"   ⚠️  No transactions found for {user_id}")
            return {
                'user_id': user_id,
                'total_transactions': 0,
                'avg_transaction_amount': 0,
                'known_devices': [],
                'known_locations': [],
                'known_merchants': [],
                'fraud_history': 0
            }
        
        # Calculate statistics
        amounts = [float(t.get('Transaction_Amount', 0)) for t in transactions]
        devices = list(set([t.get('Device_Type') for t in transactions if t.get('Device_Type')]))
        locations = list(set([t.get('Location') for t in transactions if t.get('Location')]))
        merchants = list(set([t.get('Merchant_Category') for t in transactions if t.get('Merchant_Category')]))
        fraud_count = sum([1 for t in transactions if t.get('Fraud_Label', 0) == 1])
        
        profile = {
            'user_id': user_id,
            'total_transactions': len(transactions),
            'avg_transaction_amount': sum(amounts) / len(amounts) if amounts else 0,
            'min_transaction': min(amounts) if amounts else 0,
            'max_transaction': max(amounts) if amounts else 0,
            'known_devices': devices[:10],  # Limit to top 10
            'known_locations': locations[:10],
            'known_merchants': merchants[:10],
            'fraud_history': fraud_count,
            'account_balance': float(transactions[-1].get('Account_Balance', 0)) if transactions else 0
        }
        
        print(f"   ✅ Profile loaded: {len(transactions)} transactions")
        return profile
        
    except Exception as e:
        print(f"   ❌ Error fetching profile: {e}")
        import traceback
        traceback.print_exc()
        return {
            'user_id': user_id,
            'total_transactions': 0,
            'avg_transaction_amount': 0,
            'known_devices': [],
            'known_locations': [],
            'known_merchants': [],
            'fraud_history': 0
        }


def get_user_transaction_history(user_id, days=None, limit=50):
    """
    Get user's transaction history
    OPTIMIZED: Uses query with GSI and proper limit
    """
    
    print(f"   📜 Retrieving up to {limit} transactions...")
    
    table = get_dynamodb_table()
    
    try:
        if days:
            # With date filter
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            response = table.query(
                IndexName='UserIdIndex',
                KeyConditionExpression='User_ID = :uid AND #ts >= :cutoff',
                ExpressionAttributeNames={'#ts': 'Timestamp'},
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':cutoff': cutoff_date
                },
                Limit=limit,
                ScanIndexForward=False  # Sort descending (newest first)
            )
        else:
            # No date filter - just get latest N transactions
            response = table.query(
                IndexName='UserIdIndex',
                KeyConditionExpression='User_ID = :uid',
                ExpressionAttributeValues={':uid': user_id},
                Limit=limit,
                ScanIndexForward=False  # Sort descending
            )
        
        transactions = response.get('Items', [])
        
        # Only paginate if we haven't hit the limit
        while 'LastEvaluatedKey' in response and len(transactions) < limit:
            remaining = limit - len(transactions)
            
            if days:
                response = table.query(
                    IndexName='UserIdIndex',
                    KeyConditionExpression='User_ID = :uid AND #ts >= :cutoff',
                    ExpressionAttributeNames={'#ts': 'Timestamp'},
                    ExpressionAttributeValues={
                        ':uid': user_id,
                        ':cutoff': cutoff_date
                    },
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    Limit=remaining,
                    ScanIndexForward=False
                )
            else:
                response = table.query(
                    IndexName='UserIdIndex',
                    KeyConditionExpression='User_ID = :uid',
                    ExpressionAttributeValues={':uid': user_id},
                    ExclusiveStartKey=response['LastEvaluatedKey'],
                    Limit=remaining,
                    ScanIndexForward=False
                )
            
            transactions.extend(response.get('Items', []))
        
        print(f"   ✅ Found {len(transactions)} transactions")
        return transactions
        
    except Exception as e:
        print(f"   ❌ Error fetching history: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_recent_transactions_count(user_id, hours=24):
    """
    Get count of transactions in last X hours
    OPTIMIZED: Uses query with count only
    """
    
    table = get_dynamodb_table()
    
    try:
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        response = table.query(
            IndexName='UserIdIndex',
            KeyConditionExpression='User_ID = :uid AND #ts >= :cutoff',
            ExpressionAttributeNames={'#ts': 'Timestamp'},
            ExpressionAttributeValues={
                ':uid': user_id,
                ':cutoff': cutoff_time
            },
            Select='COUNT'
        )
        
        count = response.get('Count', 0)
        
        # Handle pagination for count
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='UserIdIndex',
                KeyConditionExpression='User_ID = :uid AND #ts >= :cutoff',
                ExpressionAttributeNames={'#ts': 'Timestamp'},
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':cutoff': cutoff_time
                },
                ExclusiveStartKey=response['LastEvaluatedKey'],
                Select='COUNT'
            )
            count += response.get('Count', 0)
        
        return count
        
    except Exception as e:
        print(f"   ❌ Error counting recent transactions: {e}")
        return 0