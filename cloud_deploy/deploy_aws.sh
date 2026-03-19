#!/bin/bash
# deploy_aws.sh
# Example deployment script pushing the model containers to AWS ECS (Elastic Container Service)

set -e

AWS_REGION="us-east-1"
ACCOUNT_ID="123456789012"
API_REPO="breast-cancer-api"
UI_REPO="breast-cancer-dashboard"

echo "Logging into AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Building Docker Images..."
docker build -t $API_REPO:latest -f Dockerfile.api .
docker build -t $UI_REPO:latest -f Dockerfile.dashboard .

echo "Tagging & Pushing..."
docker tag $API_REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO:latest
docker tag $UI_REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$UI_REPO:latest

docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$UI_REPO:latest

echo "Successfully pushed! Update your ECS Task Definitions to point to the new images."
