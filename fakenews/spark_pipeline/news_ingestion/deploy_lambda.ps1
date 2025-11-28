# Deploy News Ingestion Pipeline as Lambda Function
# This creates a Lambda function that runs hourly via EventBridge

param(
    [string]$Region = "ap-southeast-2",
    [string]$FunctionName = "news-ingestion-pipeline",
    [string]$S3Bucket = "fakenews-news-database",
    [string]$DynamoDBTable = "fakenews-scraped-news",
    [string]$RoleName = "news-ingestion-lambda-role"
)

$accountId = "979207815314"
$ecrUri = "${accountId}.dkr.ecr.${Region}.amazonaws.com/news-ingestion-pipeline:latest"

Write-Host "=== DEPLOYING NEWS INGESTION PIPELINE TO AWS ===" -ForegroundColor Yellow
Write-Host ""

# Step 1: Build Docker image
Write-Host "Step 1: Building Docker image..." -ForegroundColor Cyan
Write-Host "  Building from project root to include news_scraper.py..." -ForegroundColor Gray

# Change to project root for build context
$originalDir = Get-Location
$projectRoot = Split-Path (Split-Path (Split-Path $originalDir))
Push-Location $projectRoot

$env:DOCKER_BUILDKIT = "0"
$dockerfilePath = Join-Path $originalDir "Dockerfile"
$buildOutput = docker build --platform linux/amd64 -t news-ingestion-pipeline:latest -f $dockerfilePath . 2>&1

Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed!" -ForegroundColor Red
    Write-Host "Make sure Docker Desktop is running" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Build successful!" -ForegroundColor Green
Write-Host ""

# Step 2: Login to ECR
Write-Host "Step 2: Logging into ECR..." -ForegroundColor Cyan
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ecrUri 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ECR login failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Logged in to ECR" -ForegroundColor Green
Write-Host ""

# Step 3: Create ECR repository if needed
Write-Host "Step 3: Checking ECR repository..." -ForegroundColor Cyan
$ecrCheck = aws ecr describe-repositories --repository-names news-ingestion-pipeline --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating ECR repository..." -ForegroundColor Gray
    aws ecr create-repository --repository-name news-ingestion-pipeline --region $Region | Out-Null
    Write-Host "  [OK] ECR repository created" -ForegroundColor Gray
} else {
    Write-Host "  [OK] ECR repository already exists" -ForegroundColor Gray
}
Write-Host ""

# Step 4: Tag and push image
Write-Host "Step 4: Pushing image to ECR..." -ForegroundColor Cyan
docker tag news-ingestion-pipeline:latest $ecrUri
$pushOutput = docker push $ecrUri 2>&1 | Select-Object -Last 3

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Image pushed to ECR" -ForegroundColor Green
Write-Host ""

# Step 5: Create or update Lambda function
Write-Host "Step 5: Creating/updating Lambda function..." -ForegroundColor Cyan
Start-Sleep -Seconds 5  # Wait for ECR to be ready

# Check if function exists
$functionCheck = aws lambda get-function --function-name $FunctionName --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Updating existing function..." -ForegroundColor Gray
    $updateOutput = aws lambda update-function-code `
        --function-name $FunctionName `
        --image-uri $ecrUri `
        --region $Region 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Lambda update failed!" -ForegroundColor Red
        Write-Host $updateOutput -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "  [OK] Function updated" -ForegroundColor Gray
} else {
    Write-Host "  Creating new function..." -ForegroundColor Gray
    
    # Get IAM role ARN
    $roleArn = "arn:aws:iam::${accountId}:role/${RoleName}"
    
    # Check if role exists
    $roleCheck = aws iam get-role --role-name $RoleName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] IAM role not found. Creating it..." -ForegroundColor Yellow
        .\create_iam_role.ps1 -RoleName $RoleName -Region $Region
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to create IAM role. Please run create_iam_role.ps1 manually" -ForegroundColor Red
            exit 1
        }
    }
    
    $createOutput = aws lambda create-function `
        --function-name $FunctionName `
        --package-type Image `
        --code ImageUri=$ecrUri `
        --role $roleArn `
        --timeout 900 `
        --memory-size 1024 `
        --environment "Variables={
            TRACKING_FILE=/tmp/scraper_tracking.json,
            CSV_FILE=/tmp/news_articles.csv,
            USE_S3=true,
            S3_TRACKING_PATH=s3://${S3Bucket}/tracking/scraper_tracking.json,
            S3_CSV_PATH=s3://${S3Bucket}/articles/news_articles.csv,
            S3_PARQUET_PATH=s3://${S3Bucket}/parquet/,
            DYNAMODB_TABLE=${DynamoDBTable},
            REGION=${Region}
        }" `
        --region $Region 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Lambda creation failed!" -ForegroundColor Red
        Write-Host $createOutput -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Note: You may need to create the IAM role first:" -ForegroundColor Yellow
        Write-Host "  Role: lambda-execution-role" -ForegroundColor White
        Write-Host "  Permissions: DynamoDB, S3, CloudWatch Logs" -ForegroundColor White
        exit 1
    }
    
    Write-Host "  [OK] Function created" -ForegroundColor Gray
}

Write-Host "[OK] Lambda function ready!" -ForegroundColor Green
Write-Host ""

# Step 6: Setup EventBridge scheduler
Write-Host "Step 6: Setting up EventBridge scheduler..." -ForegroundColor Cyan
python setup_scheduler.py --rule_name news-ingestion-hourly --schedule "rate(1 hour)" --lambda_function $FunctionName --region $Region

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Scheduler setup failed - you can run it manually later" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Scheduler configured" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] Lambda function: $FunctionName" -ForegroundColor Green
Write-Host "[OK] ECR image: $ecrUri" -ForegroundColor Green
Write-Host "[OK] Schedule: Every hour" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify Lambda function in AWS Console" -ForegroundColor White
Write-Host "  2. Check CloudWatch logs: /aws/lambda/$FunctionName" -ForegroundColor White
Write-Host "  3. Test manually: aws lambda invoke --function-name $FunctionName ..." -ForegroundColor White
Write-Host "  4. Monitor DynamoDB table: $DynamoDBTable" -ForegroundColor White
Write-Host ""

