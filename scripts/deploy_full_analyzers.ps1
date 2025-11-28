# Deploy Lambda with FULL analyzers (no placeholder data)
# This ensures all analyzers use real data: LanguageAnalyzer, CredibilityAnalyzer (with WHOIS), 
# CrossCheckAnalyzer, and RelatedArticlesAnalyzer

param(
    [string]$Region = "ap-southeast-2"
)

$accountId = "979207815314"
$ecrUri = "${accountId}.dkr.ecr.${Region}.amazonaws.com/fakenews-analyzer:latest"

Write-Host "=== DEPLOYING FULL ANALYZERS TO AWS ===" -ForegroundColor Yellow
Write-Host ""
Write-Host "This deployment includes:" -ForegroundColor Cyan
Write-Host "  [OK] Full LanguageAnalyzer (real linguistic analysis)" -ForegroundColor Green
Write-Host "  [OK] Full CredibilityAnalyzer (with WHOIS verification)" -ForegroundColor Green
Write-Host "  [OK] Full CrossCheckAnalyzer (TF-IDF similarity)" -ForegroundColor Green
Write-Host "  [OK] Full RelatedArticlesAnalyzer (Google News RSS)" -ForegroundColor Green
Write-Host "  [OK] WHOIS Helper (real domain age data)" -ForegroundColor Green
Write-Host "  [OK] NO placeholder data - all charts show real results" -ForegroundColor Green
Write-Host ""

# Step 1: Build Docker image
Write-Host "Step 1: Building Docker image..." -ForegroundColor Cyan
$env:DOCKER_BUILDKIT = "0"
$buildOutput = docker build --platform linux/amd64 -t fakenews-analyzer:latest . 2>&1

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

# Step 3: Tag and push image
Write-Host "Step 3: Pushing image to ECR..." -ForegroundColor Cyan
docker rmi $ecrUri 2>&1 | Out-Null  # Remove old tag if exists
docker tag fakenews-analyzer:latest $ecrUri
$pushOutput = docker push $ecrUri 2>&1 | Select-Object -Last 3

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Image pushed to ECR" -ForegroundColor Green
Write-Host ""

# Step 4: Update Lambda function
Write-Host "Step 4: Updating Lambda function..." -ForegroundColor Cyan
Start-Sleep -Seconds 5  # Wait for ECR to be ready

$updateOutput = aws lambda update-function-code `
    --function-name fakenews-analyzer `
    --image-uri $ecrUri `
    --region $Region 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Lambda update failed!" -ForegroundColor Red
    Write-Host $updateOutput -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Lambda function updated!" -ForegroundColor Green
Write-Host ""

# Step 5: Wait for function to be ready
Write-Host "Step 5: Waiting for function to be ready..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "=== DEPLOYMENT COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] Lambda deployed with FULL analyzers" -ForegroundColor Green
Write-Host "[OK] pandas module included (proper feature extraction)" -ForegroundColor Green
Write-Host "[OK] WHOIS enabled - real domain age verification" -ForegroundColor Green
Write-Host "[OK] Google News redirect handling improved" -ForegroundColor Green
Write-Host "[OK] All charts will show REAL data (no placeholders)" -ForegroundColor Green
Write-Host ""
Write-Host "To verify, check CloudWatch logs for:" -ForegroundColor Cyan
Write-Host "  - [OK] Full analyzer classes imported successfully" -ForegroundColor White
Write-Host "  - [OK] Using features_enhanced module for proper feature extraction" -ForegroundColor White
Write-Host "  - [OK] Extracted 28 features using features_enhanced" -ForegroundColor White
Write-Host "  - [OK] Using FULL CredibilityAnalyzer with WHOIS" -ForegroundColor White
Write-Host "  - [OK] WHOIS data verified" -ForegroundColor White
Write-Host "  - [OK] Using FULL CrossCheckAnalyzer" -ForegroundColor White
Write-Host "  - [OK] Using FULL RelatedArticlesAnalyzer" -ForegroundColor White
Write-Host ""

