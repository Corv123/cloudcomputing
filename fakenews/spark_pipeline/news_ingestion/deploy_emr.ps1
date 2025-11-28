# Deploy News Ingestion Pipeline as EMR Job
# This creates an EMR cluster that runs the pipeline hourly

param(
    [string]$Region = "ap-southeast-2",
    [string]$S3Bucket = "fakenews-news-database",
    [string]$DynamoDBTable = "fakenews-articles"
)

Write-Host "=== DEPLOYING NEWS INGESTION PIPELINE TO EMR ===" -ForegroundColor Yellow
Write-Host ""

# Step 1: Upload pipeline code to S3
Write-Host "Step 1: Uploading pipeline code to S3..." -ForegroundColor Cyan

$s3CodePath = "s3://${S3Bucket}/code/news-ingestion-pipeline/"

# Create zip file with pipeline code
$tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
Copy-Item -Path "*.py" -Destination $tempDir -ErrorAction SilentlyContinue
Copy-Item -Path "../../analyzers/news_scraper.py" -Destination $tempDir -ErrorAction SilentlyContinue

$zipFile = Join-Path $env:TEMP "news-ingestion-pipeline.zip"
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipFile -Force

# Upload to S3
aws s3 cp $zipFile "${s3CodePath}news-ingestion-pipeline.zip" --region $Region

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to upload code to S3" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Code uploaded to S3" -ForegroundColor Green
Write-Host ""

# Step 2: Create EMR cluster configuration
Write-Host "Step 2: Creating EMR cluster..." -ForegroundColor Cyan

$clusterName = "news-ingestion-pipeline"
$releaseLabel = "emr-6.15.0"  # Latest EMR release

# Create cluster
$clusterId = aws emr create-cluster `
    --name $clusterName `
    --release-label $releaseLabel `
    --instance-type m5.xlarge `
    --instance-count 2 `
    --applications Name=Spark Name=Hadoop `
    --ec2-attributes KeyName=your-key-name,SubnetId=subnet-xxxxx `
    --service-role EMR_DefaultRole `
    --job-flow-role EMR_EC2_DefaultRole `
    --region $Region `
    --query 'ClusterId' `
    --output text

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create EMR cluster" -ForegroundColor Red
    Write-Host "Note: You may need to configure:" -ForegroundColor Yellow
    Write-Host "  - KeyName: Your EC2 key pair" -ForegroundColor White
    Write-Host "  - SubnetId: Your VPC subnet" -ForegroundColor White
    Write-Host "  - IAM roles: EMR_DefaultRole and EMR_EC2_DefaultRole" -ForegroundColor White
    exit 1
}

Write-Host "[OK] EMR cluster created: $clusterId" -ForegroundColor Green
Write-Host ""

# Step 3: Submit Spark job
Write-Host "Step 3: Submitting Spark job..." -ForegroundColor Cyan

$stepId = aws emr add-steps `
    --cluster-id $clusterId `
    --steps "Type=spark,Name=NewsIngestionPipeline,ActionOnFailure=CONTINUE,Args=[
        --deploy-mode,cluster,
        --py-files,${s3CodePath}news-ingestion-pipeline.zip,
        ${s3CodePath}main.py,
        --tracking_file,/tmp/scraper_tracking.json,
        --csv_file,/tmp/news_articles.csv,
        --use_s3,
        --s3_tracking,s3://${S3Bucket}/tracking/scraper_tracking.json,
        --s3_csv,s3://${S3Bucket}/articles/news_articles.csv,
        --s3_parquet,s3://${S3Bucket}/parquet/,
        --dynamodb_table,${DynamoDBTable},
        --region,${Region}
    ]" `
    --region $Region `
    --query 'StepIds[0]' `
    --output text

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to submit job" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Job submitted: $stepId" -ForegroundColor Green
Write-Host ""

Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] EMR cluster: $clusterId" -ForegroundColor Green
Write-Host "[OK] Job step: $stepId" -ForegroundColor Green
Write-Host ""
Write-Host "Note: For hourly execution, use EventBridge to trigger EMR step function" -ForegroundColor Yellow
Write-Host ""

