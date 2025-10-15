import pandas as pd
import json
from collections import Counter

# ============================================================================
# Load Kaggle Dataset
# ============================================================================
df = pd.read_csv('synthetic_fraud_dataset.csv')
# Convert timestamp to datetime FIRST (before any operations)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
print(f"📂 Loaded {len(df)} transactions with {df['User_ID'].nunique()} unique users")


# ============================================================================
# PART 1: Generate User Profiles Dataset
# ============================================================================

def generate_user_profiles(transactions_df):
    """Aggregate transaction data to create user profiles"""
    
    user_profiles = []
    
    for user_id in transactions_df['User_ID'].unique():
        user_txns = transactions_df[transactions_df['User_ID'] == user_id]
        
        # Basic stats
        profile = {
            'user_id': user_id,
            'total_transactions': len(user_txns),
            'avg_transaction_amount': user_txns['Transaction_Amount'].mean(),
            'account_age_days': (user_txns['Timestamp'].max() - user_txns['Timestamp'].min()).days,
            'transaction_frequency': len(user_txns) / max((user_txns['Timestamp'].max() - user_txns['Timestamp'].min()).days, 1),
            
            # Behavioral baselines (top 3 most frequent)
            'typical_locations': Counter(user_txns['Location']).most_common(3),
            'home_location': user_txns['Location'].mode()[0] if not user_txns['Location'].mode().empty else None,
            'known_devices': user_txns['Device_Type'].unique().tolist(),
            'typical_merchants': Counter(user_txns['Merchant_Category']).most_common(3),
            
            # Risk profile
            'fraud_history_count': user_txns['Previous_Fraudulent_Activity'].sum(),
            'highest_transaction': user_txns['Transaction_Amount'].max(),
            'typical_card_type': user_txns['Card_Type'].mode()[0] if not user_txns['Card_Type'].mode().empty else None,
            
            # Temporal patterns
            'active_hours': Counter(pd.to_datetime(user_txns['Timestamp']).dt.hour).most_common(3),
            'weekend_transaction_ratio': user_txns['Is_Weekend'].mean(),
            
            # Placeholder for deepfake feature
            'reference_photo_s3_path': f"s3://fraudguard-data-dev/user-photos/reference/{user_id}.jpg" if user_txns['Transaction_Amount'].max() > 10000 else None
        }
        
        user_profiles.append(profile)
    
    return pd.DataFrame(user_profiles)

# Generate user profiles
user_profiles_df = generate_user_profiles(df)

# Save to JSON (for DynamoDB upload)
user_profiles_df.to_json('user_profiles.json', orient='records', indent=2)
print(f"✅ Generated {len(user_profiles_df)} user profiles → user_profiles.json")


# ============================================================================
# PART 2: Enhance Transaction Dataset with Calculated Features
# ============================================================================

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract temporal features
df['hour_of_day'] = df['Timestamp'].dt.hour
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['is_unusual_hour'] = df['hour_of_day'].apply(lambda x: 1 if 2 <= x <= 6 else 0)

# Calculate behavioral deviations
df['amount_deviation_ratio'] = df['Transaction_Amount'] / df['Avg_Transaction_Amount_7d'].replace(0, 1)

# Create user profile lookup for anomaly detection
user_max_amounts = df.groupby('User_ID')['Transaction_Amount'].max().to_dict()
user_devices = df.groupby('User_ID')['Device_Type'].apply(list).to_dict()

# Anomaly flags
df['is_high_value'] = df.apply(lambda row: 1 if row['Transaction_Amount'] > user_max_amounts.get(row['User_ID'], 0) * 0.8 else 0, axis=1)
df['is_new_device'] = df.apply(lambda row: 0 if row['Device_Type'] in user_devices.get(row['User_ID'], []) else 1, axis=1)

# Save enhanced transaction dataset
df.to_csv('transactions_enhanced.csv', index=False)
print(f"✅ Enhanced {len(df)} transactions → transactions_enhanced.csv")

# ============================================================================
# Feature Summary
# ============================================================================
print("\n📊 FINAL FEATURE SUMMARY:")
print(f"User Profiles: {len(user_profiles_df.columns)} attributes")
print(f"Transaction Features: {len(df.columns)} columns")
print(f"\nNew calculated features:")
print("  - hour_of_day")
print("  - day_of_week")
print("  - is_unusual_hour")
print("  - amount_deviation_ratio")
print("  - is_high_value")
print("  - is_new_device")

# ============================================================================
# PART 3: Create Demo Scenarios (5 Hero Examples)
# ============================================================================

demo_scenarios = []

# Scenario 1: Clean legitimate
clean = df[(df['Fraud_Label'] == 0) & (df['Risk_Score'] < 20)].iloc[0].to_dict()
demo_scenarios.append({'scenario_name': 'Clean Transaction', **clean})

# Scenario 2: Velocity attack
velocity = df[(df['Fraud_Label'] == 1) & (df['Daily_Transaction_Count'] > 10)].iloc[0].to_dict()
demo_scenarios.append({'scenario_name': 'Velocity Attack', **velocity})

# Scenario 3: Deepfake (augment with deepfake attributes)
deepfake = df[(df['Fraud_Label'] == 1) & (df['Transaction_Amount'] > 10000)].iloc[0].to_dict()
deepfake['scenario_name'] = 'Deepfake Account Takeover'
deepfake['uploaded_photo_s3_path'] = 's3://fraudguard-data-dev/user-photos/uploads/deepfake_demo.jpg'
deepfake['biometric_match_score'] = 0.43
deepfake['deepfake_confidence'] = 0.92
demo_scenarios.append(deepfake)

# Scenario 4: Amount anomaly
amount_anomaly = df[(df['Fraud_Label'] == 1) & (df['amount_deviation_ratio'] > 10)].iloc[0].to_dict()
demo_scenarios.append({'scenario_name': 'Amount Anomaly', **amount_anomaly})

# Scenario 5: Geographic impossibility
geo_impossible = df[(df['Fraud_Label'] == 1) & (df['Transaction_Distance'] > 1000)].iloc[0].to_dict()
demo_scenarios.append({'scenario_name': 'Geographic Impossibility', **geo_impossible})

# Save demo scenarios
with open('demo_scenarios.json', 'w') as f:
    json.dump(demo_scenarios, f, indent=2, default=str)
print(f"\n✅ Created 5 demo scenarios → demo_scenarios.json")

print("\n🎉 Data generation complete!")