import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pickle
import json

print("🚀 Starting XGBoost Fraud Detection Model Training...\n")

# ============================================================================
# STEP 1: Load Enhanced Transaction Dataset
# ============================================================================
print("📂 Loading data...")
df = pd.read_csv('transactions_enhanced.csv')
print(f"✅ Loaded {len(df)} transactions")
print(f"   - Fraud cases: {df['Fraud_Label'].sum()} ({df['Fraud_Label'].mean()*100:.1f}%)")
print(f"   - Legitimate: {(df['Fraud_Label']==0).sum()} ({(df['Fraud_Label']==0).mean()*100:.1f}%)\n")

# ============================================================================
# STEP 2: Feature Selection (22 features for training)
# ============================================================================
print("🎯 Selecting features...")

# Features to EXCLUDE from training
exclude_features = [
    'Transaction_ID',      # Identifier
    'User_ID',            # Identifier
    'Timestamp',          # Already extracted to features
    'Fraud_Label',        # Target variable
    'Risk_Score'          # Optional: pre-computed score
]

# Get all feature columns
feature_cols = [col for col in df.columns if col not in exclude_features]
print(f"✅ Using {len(feature_cols)} features for training\n")

# ============================================================================
# STEP 3: Encode Categorical Features
# ============================================================================
print("🔧 Encoding categorical features...")

categorical_features = [
    'Transaction_Type', 
    'Device_Type', 
    'Location', 
    'Merchant_Category', 
    'Card_Type', 
    'Authentication_Method'
]

# Store encoders for later use in inference
encoders = {}

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

print(f"✅ Encoded {len(encoders)} categorical features\n")

# Save encoders
with open('model-artifacts/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("💾 Saved label encoders → label_encoders.pkl\n")

# ============================================================================
# STEP 4: Prepare Training Data
# ============================================================================
print("📊 Preparing train/val/test split...")

X = df[feature_cols]
y = df['Fraud_Label']

# Stratified split (maintains fraud ratio in each set)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 0.85 ≈ 0.15 of total
)

print(f"✅ Train: {len(X_train)} ({y_train.sum()} fraud)")
print(f"✅ Val:   {len(X_val)} ({y_val.sum()} fraud)")
print(f"✅ Test:  {len(X_test)} ({y_test.sum()} fraud)\n")

# ============================================================================
# STEP 5: Train XGBoost Model
# ============================================================================
print("🎓 Training XGBoost model...")

# Calculate scale_pos_weight for imbalanced data
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    max_depth=6,                    # Prevent overfitting
    learning_rate=0.1,              # Standard learning rate
    n_estimators=150,               # Number of trees
    scale_pos_weight=fraud_ratio,   # Handle class imbalance
    eval_metric='auc',              # Optimize for AUC
    random_state=42,
    use_label_encoder=False
)

# Train with validation set for early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("✅ Model training complete!\n")

# ============================================================================
# STEP 6: Evaluate Model
# ============================================================================
print("📈 Evaluating model performance...\n")

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.35).astype(int)

# Metrics
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"🎯 AUC-ROC Score: {auc_score:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

print("🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives:  {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives:  {cm[1][1]}\n")

# ============================================================================
# STEP 7: Feature Importance
# ============================================================================
print("🔑 Top 10 Most Important Features:")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<30} {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('model-artifacts/feature_importance.csv', index=False)
print("\n💾 Saved feature importance → feature_importance.csv\n")

# ============================================================================
# STEP 8: Save Model
# ============================================================================
print("💾 Saving trained model...")

# Save as pickle
with open('model-artifacts/xgboost_fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved → xgboost_fraud_model.pkl")

# Save model metadata
metadata = {
    'features': feature_cols,
    'n_features': len(feature_cols),
    'auc_score': float(auc_score),
    'fraud_ratio': float(fraud_ratio),
    'training_samples': len(X_train),
    'categorical_features': categorical_features
}

with open('model-artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata saved → model_metadata.json\n")

# ============================================================================
# STEP 9: Test Inference Function
# ============================================================================
print("🧪 Testing inference on sample transaction...")

def predict_fraud(transaction_data):
    """Test inference function"""
    fraud_prob = model.predict_proba([transaction_data])[0][1]
    risk_score = int(fraud_prob * 100)
    prediction = 'FRAUD' if fraud_prob >= 0.5 else 'LEGITIMATE'
    return {
        'fraud_probability': fraud_prob,
        'risk_score': risk_score,
        'prediction': prediction
    }

# Test with first test sample
sample = X_test.iloc[0].values
result = predict_fraud(sample)
print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
print(f"   Risk Score: {result['risk_score']}/100")
print(f"   Prediction: {result['prediction']}")
print(f"   Actual Label: {'FRAUD' if y_test.iloc[0] == 1 else 'LEGITIMATE'}\n")

print("🎉 Model training complete! Ready for deployment.")
print("\n📦 Generated files:")
print("   ✅ xgboost_fraud_model.pkl    - Trained model")
print("   ✅ label_encoders.pkl         - Categorical encoders")
print("   ✅ model_metadata.json        - Model info")
print("   ✅ feature_importance.csv     - Feature rankings")