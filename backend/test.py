import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Testing AWS Bedrock Connection...\n")

# Load credentials
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION', 'us-east-1')

print(f"AWS Region: {region}")
print(f"Access Key: {access_key[:10]}... (masked)")
print(f"Secret Key: {'*' * 20}\n")

# Test 1: Check credentials
print("Test 1: Checking AWS credentials...")
try:
    sts = boto3.client('sts', 
                       aws_access_key_id=access_key,
                       aws_secret_access_key=secret_key,
                       region_name=region)
    
    identity = sts.get_caller_identity()
    print(f"✅ Credentials valid!")
    print(f"   Account: {identity['Account']}")
    print(f"   User ARN: {identity['Arn']}\n")
except Exception as e:
    print(f"❌ Credentials invalid: {e}\n")
    exit(1)

# Test 2: List available Bedrock models
print("Test 2: Listing available Bedrock models in your region...")
try:
    bedrock = boto3.client('bedrock',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key,
                          region_name=region)
    
    response = bedrock.list_foundation_models()
    
    claude_models = [m for m in response['modelSummaries'] if 'claude' in m['modelId'].lower()]
    
    if claude_models:
        print(f"✅ Found {len(claude_models)} Claude models:\n")
        for model in claude_models:
            print(f"   - {model['modelId']}")
            print(f"     Status: {model.get('modelLifecycle', {}).get('status', 'Unknown')}")
    else:
        print("⚠️  No Claude models found in this region")
        print("   Try switching to us-east-1 or us-west-2\n")
    
    print()
    
except Exception as e:
    print(f"❌ Cannot list models: {e}")
    print("   This might be a permissions issue\n")

# Test 3: Try invoking a model
print("Test 3: Testing model invocation...")

# Try common model IDs
test_models = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-v2:1"
]

bedrock_runtime = boto3.client('bedrock-runtime',
                              aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key,
                              region_name=region)

for model_id in test_models:
    print(f"\n   Trying: {model_id}")
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say 'Hello' if you can read this."}]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body['content'][0]['text']
        
        print(f"   ✅ SUCCESS! Model works!")
        print(f"   Response: {result}")
        print(f"\n🎉 Use this model ID in your .env:")
        print(f"   BEDROCK_MODEL_ID={model_id}")
        break
        
    except Exception as e:
        error_msg = str(e)
        if "ResourceNotFoundException" in error_msg or "model identifier is invalid" in error_msg:
            print(f"   ❌ Model not available in {region}")
        elif "AccessDeniedException" in error_msg:
            print(f"   ❌ No permission to access this model")
        else:
            print(f"   ❌ Error: {error_msg}")

print("\n✅ Test complete!")