# Cleanup failed SageMaker resources

param(
    [string]$ModelName = "fakenews-sensationalism-model",
    [string]$EndpointName = "fakenews-sensationalism-endpoint",
    [string]$Region = "ap-southeast-2"
)

Write-Host "=== Cleaning Up Failed SageMaker Resources ===" -ForegroundColor Green
Write-Host ""

# Delete endpoint if exists
Write-Host "Checking for endpoint: $EndpointName..." -ForegroundColor Yellow
$endpointCheck = aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Deleting endpoint..." -ForegroundColor Gray
    aws sagemaker delete-endpoint --endpoint-name $EndpointName --region $Region | Out-Null
    Write-Host "  [OK] Endpoint deletion initiated" -ForegroundColor Gray
} else {
    Write-Host "  Endpoint not found (or already deleted)" -ForegroundColor Gray
}

# Delete endpoint config if exists
$configName = "$EndpointName-config"
Write-Host "Checking for endpoint config: $configName..." -ForegroundColor Yellow
$configCheck = aws sagemaker describe-endpoint-config --endpoint-config-name $configName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Deleting endpoint config..." -ForegroundColor Gray
    aws sagemaker delete-endpoint-config --endpoint-config-name $configName --region $Region | Out-Null
    Write-Host "  [OK] Endpoint config deleted" -ForegroundColor Gray
} else {
    Write-Host "  Endpoint config not found (or already deleted)" -ForegroundColor Gray
}

# Delete model if exists
Write-Host "Checking for model: $ModelName..." -ForegroundColor Yellow
$modelCheck = aws sagemaker describe-model --model-name $ModelName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Deleting model..." -ForegroundColor Gray
    aws sagemaker delete-model --model-name $ModelName --region $Region | Out-Null
    Write-Host "  [OK] Model deleted" -ForegroundColor Gray
} else {
    Write-Host "  Model not found (or already deleted)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Green

