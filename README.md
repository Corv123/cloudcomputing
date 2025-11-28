# Fake News Detector

A cloud-based fake news detection system that analyzes news articles for credibility using machine learning and big data technologies.

## Already Deployed on AWS 
https://d1wj1cwvm99kgr.cloudfront.net/

## Overview

The Fake News Detector is a serverless application built on AWS that provides real-time credibility analysis of news articles. The system combines distributed big data processing (Apache Spark) for model training with serverless inference (AWS Lambda) for low-latency analysis.

## Architecture

- **Frontend**: Static website hosted on S3, distributed via CloudFront
- **Backend**: AWS Lambda functions for real-time article analysis
- **Data Storage**: DynamoDB for article metadata, S3 for models and datasets
- **Big Data**: AWS EMR with Apache Spark for distributed model training
- **API**: API Gateway RESTful API

## Key Features

- **Real-time Analysis**: Analyze news articles in seconds
- **Credibility Scoring**: Multi-factor credibility assessment
- **Sentiment Analysis**: VADER sentiment analysis
- **Cross-Check**: Compare articles against verified database
- **Language Quality**: Assess writing quality and grammar
- **Interactive UI**: Glassmorphism design with interactive charts
- **Scalable**: Handles thousands to millions of articles

## Project Structure

```
.
├── fakenews/                    # Main application code
│   ├── analyzers/               # Analysis modules
│   ├── static/                  # Frontend (HTML, CSS, JS)
│   ├── spark_pipeline/           # Big data training pipeline
│   ├── models/                  # Trained ML models
│   └── datasets/                # Training datasets
├── lambda_function.py           # AWS Lambda handler
├── Dockerfile                   # Lambda container image
├── scripts/                     # Deployment scripts
└── config/                      # AWS configuration files
```

## Setup

### Prerequisites

- Python 3.11+
- AWS CLI configured
- Docker (for Lambda deployment)
- PowerShell (for deployment scripts)

### Local Development

1. Install dependencies:
```bash
pip install -r fakenews/requirements.txt
```

2. Set up environment variables (see `config/lambda-env-vars.json`)

3. Run locally:
```bash
cd fakenews
python app.py
```

## Deployment

### Deploy Frontend
```powershell
.\scripts\deploy_frontend.ps1
```

### Deploy Lambda Function
```powershell
# Build and push Docker image to ECR
docker build -t fakenews-analyzer .
docker tag fakenews-analyzer:latest <account-id>.dkr.ecr.ap-southeast-2.amazonaws.com/fakenews-analyzer:latest
docker push <account-id>.dkr.ecr.ap-southeast-2.amazonaws.com/fakenews-analyzer:latest

# Update Lambda function
aws lambda update-function-code --function-name fakenews-analyzer --image-uri <account-id>.dkr.ecr.ap-southeast-2.amazonaws.com/fakenews-analyzer:latest
```

### Deploy SageMaker Endpoint (Optional)
```powershell
.\scripts\build_and_deploy_sagemaker_custom.ps1
```

## Usage

1. Open the frontend URL (CloudFront distribution)
2. Enter a news article URL
3. Click "Analyze Article"
4. View credibility breakdown, sentiment analysis, and cross-check results

## Big Data Pipeline

The Spark pipeline processes training data and generates ML models:

1. **Data Ingestion**: Load datasets from S3
2. **Data Cleaning**: Remove duplicates, handle missing values
3. **Feature Engineering**: TF-IDF vectors + linguistic features
4. **Model Training**: Distributed LinearSVC training
5. **Model Export**: Export to scikit-learn format for Lambda

Run the pipeline on EMR:
```bash
# Submit Spark job to EMR cluster
spark-submit --master yarn fakenews/spark_pipeline/01_data_ingestion.py
```

## Technologies

- **AWS Services**: Lambda, API Gateway, DynamoDB, S3, EMR, CloudFront, SageMaker
- **ML Libraries**: scikit-learn, NLTK, VADER Sentiment
- **Big Data**: Apache Spark, PySpark
- **Frontend**: HTML5, CSS3, JavaScript, Plotly.js
- **Container**: Docker

## Cost

- Low traffic (1,000 analyses/month): ~$1/month
- Medium traffic (10,000 analyses/month): ~$6.50/month
- High traffic (100,000 analyses/month): ~$63/month

## Contributors

- Corvan Chua
- Hayden Chua Shao En
- Jeanie Cherie Chua Yue-Ning
- Goh Jing Wen
- Ang Xuan Yu Pamela

