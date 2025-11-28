# PowerShell script for EMR Deployment

param(
    [string]$S3Bucket = "fakenews-ml-pipeline",
    [string]$ClusterName = "fakenews-spark-training",
    [string]$Region = "ap-southeast-2",
    [string]$InstanceType = "m5.xlarge",
    [int]$InstanceCount = 3
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AWS EMR Deployment - Spark ML Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check AWS CLI
try {
    $awsVersion = aws --version
    Write-Host "✓ AWS CLI is installed: $awsVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ AWS CLI is not installed!" -ForegroundColor Red
    Write-Host "Please install AWS CLI: https://aws.amazon.com/cli/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if bucket exists
Write-Host "Checking S3 bucket: s3://$S3Bucket" -ForegroundColor Yellow
try {
    aws s3 ls "s3://$S3Bucket" | Out-Null
    Write-Host "✓ Bucket exists" -ForegroundColor Green
} catch {
    Write-Host "Creating S3 bucket: s3://$S3Bucket" -ForegroundColor Yellow
    aws s3 mb "s3://$S3Bucket" --region $Region
    Write-Host "✓ Bucket created" -ForegroundColor Green
}

Write-Host ""

# Upload code to S3
Write-Host "Uploading pipeline code to S3..." -ForegroundColor Yellow
$codeS3Path = "s3://$S3Bucket/spark_pipeline/"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

aws s3 sync $scriptDir $codeS3Path --exclude "*.pyc" --exclude "__pycache__" --exclude "output/*"
Write-Host "✓ Code uploaded to $codeS3Path" -ForegroundColor Green
Write-Host ""

# Upload datasets to S3 (if local datasets exist)
$datasetsDir = Join-Path (Split-Path -Parent $scriptDir) "datasets"
if (Test-Path $datasetsDir) {
    Write-Host "Uploading datasets to S3..." -ForegroundColor Yellow
    $datasetsS3Path = "s3://$S3Bucket/raw-data/"
    aws s3 sync $datasetsDir $datasetsS3Path
    Write-Host "✓ Datasets uploaded to $datasetsS3Path" -ForegroundColor Green
    Write-Host ""
}

# Create EMR cluster
Write-Host "Creating EMR cluster..." -ForegroundColor Yellow
Write-Host "  Cluster Name: $ClusterName" -ForegroundColor White
Write-Host "  Instance Type: $InstanceType" -ForegroundColor White
Write-Host "  Instance Count: $InstanceCount" -ForegroundColor White
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host ""

$stepArgs = @(
    "--deploy-mode", "cluster",
    "--py-files", "$codeS3Path`main.py",
    "$codeS3Path`main.py",
    "--use_s3",
    "--s3_bucket", "s3://$S3Bucket/",
    "--datasets_dir", "raw-data/",
    "--output_dir", "output/"
)

$stepArgsJson = ($stepArgs | ConvertTo-Json -Compress).Replace('"', '\"')

$clusterId = aws emr create-cluster `
    --name $ClusterName `
    --release-label emr-6.15.0 `
    --applications Name=Spark Name=Hadoop `
    --instance-type $InstanceType `
    --instance-count $InstanceCount `
    --service-role EMR_DefaultRole `
    --ec2-attributes InstanceProfile=EMR_EC2_DefaultRole `
    --region $Region `
    --log-uri "s3://$S3Bucket/logs/" `
    --steps Type=spark,Name=TrainSensationalismModel,Args=$stepArgsJson,ActionOnFailure=TERMINATE_CLUSTER `
    --auto-terminate `
    --query 'ClusterId' `
    --output text

if (-not $clusterId) {
    Write-Host "❌ Failed to create EMR cluster!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ EMR cluster created: $clusterId" -ForegroundColor Green
Write-Host ""
Write-Host "Cluster will automatically terminate after job completion." -ForegroundColor Yellow
Write-Host "Monitor progress:" -ForegroundColor Cyan
Write-Host "  aws emr describe-cluster --cluster-id $clusterId --region $Region" -ForegroundColor White
Write-Host ""
Write-Host "View logs:" -ForegroundColor Cyan
Write-Host "  aws s3 ls s3://$S3Bucket/logs/$clusterId/" -ForegroundColor White
Write-Host ""

