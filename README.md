[README.md](https://github.com/user-attachments/files/23057751/README.md)

# FraudGuard AI — Deepfake Detection & Explainable Fraud Analysis

[![AWS](https://img.shields.io/badge/AWS-Ready-orange)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19.1.1-blue)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)

FraudGuard AI is an AI-driven platform for detecting payment fraud and deepfake-based identity spoofing. It combines machine learning, computer vision, and LLM-assisted explainability to provide auditable risk scores and CSR-friendly explanations.

## Project Gallery
<div style="display: flex; justify-content: center;">
  <img width="400" height="400" style="margin-right: 10px alt="image1" src="https://github.com/user-attachments/assets/d452d388-5a87-4e66-b3fe-f7eca6303232" />
  <img width="400" height="400" style="margin-right: 10px alt="image2" src="https://github.com/user-attachments/assets/507852f0-f747-45d9-b47c-53280f87e4e6" />
  <img width="400" height="400" style="margin-right: 10px alt="image3" src="https://github.com/user-attachments/assets/d2fc9b58-0061-49ad-bdb2-c9bb707470e9" />
  <img width="400" height="400" style="margin-right: 10px alt="image4" src="https://github.com/user-attachments/assets/0cfee5b8-0456-4038-bd5d-d8fbbdf2819e" />
</div>


## Architecture (high level)

- Frontend: React (Vite) UI for bank staff and integrations
- API Layer: Lambda functions or API Gateway for ingestion and orchestration
- Backend: FastAPI service (ECS / Fargate or local for development)
- Agents:
  - Transaction Monitoring (XGBoost)
  - Deepfake Detection (SageMaker computer vision)
  - Evidence Collector Agent (Gathers User's Historical Transaction Data to study patterns)
  - Risk Assessment (Combines Insights of first 3 agents)
  - Escalation Handler Agent (Escalates the the fraudulent transaction reports)
- Storage & Infra: S3 (photos, models, logs), DynamoDB (transactions), SageMaker, Amazon Bedrock (LLM access)
- Observability: CloudWatch, CloudTrail, X-Ray

## Key Features

- ML-based fraud scoring (XGBoost)
- Biometric deepfake detection (SageMaker)
- Explainability: SHAP, LIME, SageMaker Clarify
- Human-readable recommendations for CSRs using LLM summarization
- Modular multi-agent orchestration for extensibility and auditability

## Quick start — Minimal / no CloudFormation path

This README focuses on a minimal manual/CLI deployment so you can get running quickly. If you prefer IaC later (CDK/SAM/Terraform), that can be provided.

Prerequisites
- AWS account with appropriate permissions
- AWS CLI configured (aws configure)
- Node.js 18+ (frontend)
- Python 3.11+ (backend)
- Docker (optional — for building/pushing backend image)
- (Optional) Access to Amazon Bedrock if you need LLM capabilities

1) Clone repository
```bash
git clone <repository-url>
cd fraudguard_ai-main
```

2) Backend (local development)
- Create and activate virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
- Copy example env and edit:
```bash
cp .env.example .env
# edit .env to set local/testing values (DynamoDB local or AWS creds)
```
- Run locally:
```bash
python main.py
```
Running local backend is recommended during early development for faster iteration.

3) Frontend (local development)
```bash
cd fraudguard-frontend
npm install
npm run dev
```

4) Minimal AWS resources — CLI/Console steps

Create S3 buckets
```bash
aws s3api create-bucket --bucket <env>-fraudguard-frontend-<account-id> --region <region> --create-bucket-configuration LocationConstraint=<region>
aws s3api create-bucket --bucket <env>-fraudguard-photos-<account-id> --region <region> --create-bucket-configuration LocationConstraint=<region>
aws s3api create-bucket --bucket <env>-fraudguard-models-<account-id> --region <region> --create-bucket-configuration LocationConstraint=<region>
```

Create DynamoDB table
```bash
aws dynamodb create-table \
  --table-name <env>-fraudguard-transactions \
  --attribute-definitions AttributeName=transaction_id,AttributeType=S \
  --key-schema AttributeName=transaction_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region <region>
```

Create ECR repo (if using ECS/ECR)
```bash
aws ecr create-repository --repository-name fraudguard-backend-<env> --region <region>
```
Build and push Docker image (optional)
```bash
docker build -t fraudguard-backend .
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag fraudguard-backend:latest <account>.dkr.ecr.<region>.amazonaws.com/fraudguard-backend-<env>:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/fraudguard-backend-<env>:latest
```

Create IAM roles
- Create a Lambda execution role with:
  - AWSLambdaBasicExecutionRole
  - AmazonDynamoDBFullAccess (or scoped)
  - AmazonS3FullAccess (or scoped)
  - AmazonSageMakerFullAccess (if using SageMaker)
For production, restrict to least-privilege.

Create Lambdas (or use Lambda Function URLs for quick testing)
```bash
# Example create-function pointing to code in S3
aws lambda create-function \
  --function-name fraudguard-submit-transaction-<env> \
  --runtime python3.11 \
  --role arn:aws:iam::<account-id>:role/<lambda-exec-role> \
  --handler fraudguard_ai_submit_transaction.handler \
  --code S3Bucket=<bucket>,S3Key=fraudguard-submit-transaction.zip \
  --timeout 30
# Optional: enable Function URL for quick testing (no API Gateway)
aws lambda create-function-url-config --function-name fraudguard-submit-transaction-<env> --auth-type NONE
```

SageMaker (optional)
- If you have model artifacts and an inference image, create a SageMaker Model, EndpointConfig, and Endpoint. These are optional for initial testing (you can stub deepfake detection).

Bedrock (LLM) note
- Amazon Bedrock often requires account access and permissions. Add Bedrock model identifier to backend environment once your account has access (example: amazon.nova-pro-v1:0).

Frontend deployment (static)
```bash
cd fraudguard-frontend
npm run build
aws s3 sync dist/ s3://<env>-fraudguard-frontend-<account-id>/ --delete
# For production, use CloudFront in front of S3 and enable WAF as needed
```

Common environment variables
- DYNAMODB_TRANSACTIONS_TABLE=<env>-fraudguard-transactions
- ECS_BACKEND_URL=<backend-url> (if using ECS)
- S3_BUCKET_NAME=<env>-fraudguard-photos
- AWS_REGION=<region>
- BEDROCK_MODEL_ID=amazon.nova-pro-v1:0 (set only after Bedrock access enabled)
- SAGEMAKER_ENDPOINT_NAME=fraudguard-sagemaker-endpoint-<env>

## Data & Models
- Synthetic fraud dataset (10k+ transactions) used for training.
- XGBoost fraud model with SHAP/LIME explainers.
- SageMaker model artifacts for deepfake detection (optional).

## Security & Compliance
- Use IAM least-privilege.
- Enable server-side encryption for S3 and encryption at rest for DynamoDB where needed.
- Enable CloudTrail, CloudWatch Logs, and X-Ray for observability.
- Follow GDPR/PCI guidelines for handling personal and payment data.

## Monitoring & Observability
- CloudWatch for logs and metrics.
- X-Ray for tracing distributed requests.
- Custom dashboards for fraud metrics and alerts.




## Next steps 
- Provide scripts (bash) to automate the CLI steps above.
- Create IaC (CDK, SAM, or Terraform) for repeatable deployments.
- Harden IAM policies and add VPC endpoints for S3/DynamoDB.
