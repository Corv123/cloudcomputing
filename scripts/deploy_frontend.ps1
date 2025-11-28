# Deploy Frontend to AWS S3 and CloudFront
param(
    [string]$S3Bucket = "fakenews-detector-frontend",
    [string]$Region = "ap-southeast-2",
    [string]$CloudFrontDistributionId = "E2MU3LYFLK146H"
)

Write-Host "=== DEPLOYING FRONTEND TO AWS ===" -ForegroundColor Cyan
Write-Host ""

# Check AWS CLI
try {
    $null = aws --version 2>&1
    Write-Host "[OK] AWS CLI found" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] AWS CLI not found" -ForegroundColor Red
    exit 1
}

# Check credentials
try {
    $null = aws sts get-caller-identity --region $Region 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] AWS credentials not configured" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] AWS credentials configured" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Error checking AWS credentials" -ForegroundColor Red
    exit 1
}

# Check S3 bucket
Write-Host "Checking S3 bucket: $S3Bucket" -ForegroundColor Yellow
$bucketCheck = aws s3 ls "s3://$S3Bucket" --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Bucket might not exist. Attempting to create..." -ForegroundColor Yellow
    aws s3 mb "s3://$S3Bucket" --region $Region 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Bucket created" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to create bucket" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[OK] Bucket exists" -ForegroundColor Green
}

# Upload files
Write-Host ""
Write-Host "Uploading frontend files..." -ForegroundColor Yellow
$staticPath = "fakenews/static"

if (-not (Test-Path $staticPath)) {
    Write-Host "[ERROR] Directory not found: $staticPath" -ForegroundColor Red
    exit 1
}

# Sync all files to S3
Write-Host "Syncing files to S3..." -ForegroundColor Cyan
aws s3 sync "$staticPath" "s3://$S3Bucket" --region $Region --delete 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Files uploaded successfully" -ForegroundColor Green
} else {
    Write-Host "[WARN] Upload completed with warnings" -ForegroundColor Yellow
}

# Invalidate CloudFront cache
Write-Host ""
Write-Host "Invalidating CloudFront cache..." -ForegroundColor Yellow
$invalidationOutput = aws cloudfront create-invalidation --distribution-id $CloudFrontDistributionId --paths "/*" 2>&1
$invalidation = $invalidationOutput | ConvertFrom-Json

if ($invalidation.Invalidation) {
    Write-Host "[OK] CloudFront cache invalidation created" -ForegroundColor Green
    Write-Host "Invalidation ID: $($invalidation.Invalidation.Id)" -ForegroundColor Gray
    Write-Host "Status: $($invalidation.Invalidation.Status)" -ForegroundColor Gray
} else {
    Write-Host "[WARN] Could not create invalidation" -ForegroundColor Yellow
    Write-Host "Error output: $invalidationOutput" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host "S3 Bucket: s3://$S3Bucket" -ForegroundColor White
Write-Host "Region: $Region" -ForegroundColor White
Write-Host "Your new design should be live!" -ForegroundColor Green
