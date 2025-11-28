#!/bin/bash
# EMR Deployment Script for Spark ML Pipeline

set -e

echo "=========================================="
echo "AWS EMR Deployment - Spark ML Pipeline"
echo "=========================================="
echo ""

# Configuration
S3_BUCKET="${S3_BUCKET:-fakenews-ml-pipeline}"
CLUSTER_NAME="${CLUSTER_NAME:-fakenews-spark-training}"
REGION="${AWS_REGION:-ap-southeast-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-m5.xlarge}"
INSTANCE_COUNT="${INSTANCE_COUNT:-3}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed!"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi

echo "✓ AWS CLI is installed"
echo ""

# Check if bucket exists, create if not
echo "Checking S3 bucket: s3://${S3_BUCKET}"
if ! aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo "Creating S3 bucket: s3://${S3_BUCKET}"
    aws s3 mb "s3://${S3_BUCKET}" --region "${REGION}"
    echo "✓ Bucket created"
else
    echo "✓ Bucket exists"
fi
echo ""

# Upload code to S3
echo "Uploading pipeline code to S3..."
CODE_S3_PATH="s3://${S3_BUCKET}/spark_pipeline/"
aws s3 sync . "${CODE_S3_PATH}" --exclude "*.pyc" --exclude "__pycache__" --exclude "output/*"
echo "✓ Code uploaded to ${CODE_S3_PATH}"
echo ""

# Upload datasets to S3 (if local datasets exist)
if [ -d "../datasets" ]; then
    echo "Uploading datasets to S3..."
    DATASETS_S3_PATH="s3://${S3_BUCKET}/raw-data/"
    aws s3 sync ../datasets/ "${DATASETS_S3_PATH}"
    echo "✓ Datasets uploaded to ${DATASETS_S3_PATH}"
    echo ""
fi

# Create EMR cluster
echo "Creating EMR cluster..."
echo "  Cluster Name: ${CLUSTER_NAME}"
echo "  Instance Type: ${INSTANCE_TYPE}"
echo "  Instance Count: ${INSTANCE_COUNT}"
echo "  Region: ${REGION}"
echo ""

CLUSTER_ID=$(aws emr create-cluster \
    --name "${CLUSTER_NAME}" \
    --release-label emr-6.15.0 \
    --applications Name=Spark Name=Hadoop \
    --instance-type "${INSTANCE_TYPE}" \
    --instance-count "${INSTANCE_COUNT}" \
    --service-role EMR_DefaultRole \
    --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole \
    --region "${REGION}" \
    --log-uri "s3://${S3_BUCKET}/logs/" \
    --steps Type=spark,Name=TrainSensationalismModel,Args=[--deploy-mode,cluster,--py-files,"${CODE_S3_PATH}main.py","${CODE_S3_PATH}main.py","--use_s3","--s3_bucket","s3://${S3_BUCKET}/","--datasets_dir","raw-data/","--output_dir","output/"],ActionOnFailure=TERMINATE_CLUSTER \
    --auto-terminate \
    --query 'ClusterId' \
    --output text)

if [ -z "$CLUSTER_ID" ]; then
    echo "❌ Failed to create EMR cluster!"
    exit 1
fi

echo "✓ EMR cluster created: ${CLUSTER_ID}"
echo ""
echo "Cluster will automatically terminate after job completion."
echo "Monitor progress:"
echo "  aws emr describe-cluster --cluster-id ${CLUSTER_ID} --region ${REGION}"
echo ""
echo "View logs:"
echo "  aws s3 ls s3://${S3_BUCKET}/logs/${CLUSTER_ID}/"
echo ""

