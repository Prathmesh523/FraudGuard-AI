import boto3
import json
from config import AWS_REGION, NOVA_INFERENCE_ARN


def call_nova(prompt, max_tokens=1000, temperature=0.3):
    """
    Call Amazon Nova Pro via Bedrock
    
    Args:
        prompt (str): Text prompt for Nova
        max_tokens (int): Maximum tokens in response
        temperature (float): Sampling temperature
        
    Returns:
        str: Nova's text response
    """
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=NOVA_INFERENCE_ARN,
            body=json.dumps({
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature
                }
            }),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        result = json.loads(response["body"].read().decode("utf-8"))
        
        # Extract text from Nova response
        return result["output"]["message"]["content"][0]["text"]
        
    except Exception as e:
        print(f"Nova API Error: {e}")
        raise