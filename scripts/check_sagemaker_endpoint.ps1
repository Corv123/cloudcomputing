# Script to check SageMaker endpoint status
param(
    [Parameter(Mandatory=$false)]
    [string]$EndpointName = "fakenews-sensationalism-endpoint",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "ap-southeast-2"
)

Write-Host "=== Checking SageMaker Endpoint Status ===" -ForegroundColor Cyan
Write-Host ""

$endpoint = aws sagemaker describe-endpoint `
    --endpoint-name $EndpointName `
    --region $Region `
    --output json 2>&1 | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to describe endpoint" -ForegroundColor Red
    exit 1
}

$status = $endpoint.EndpointStatus
$creationTime = $endpoint.CreationTime
$lastModified = $endpoint.LastModifiedTime

Write-Host "Endpoint Name: $($endpoint.EndpointName)" -ForegroundColor White
Write-Host "Status: " -NoNewline

switch ($status) {
    "Creating" {
        Write-Host "$status" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The endpoint is still being created. This typically takes 5-15 minutes." -ForegroundColor Gray
        Write-Host "SageMaker is:" -ForegroundColor Gray
        Write-Host "  - Provisioning the instance (ml.t2.medium)" -ForegroundColor DarkGray
        Write-Host "  - Pulling the Docker image from ECR" -ForegroundColor DarkGray
        Write-Host "  - Starting the container" -ForegroundColor DarkGray
        Write-Host "  - Loading the model" -ForegroundColor DarkGray
        Write-Host "  - Running health checks" -ForegroundColor DarkGray
    }
    "InService" {
        Write-Host "$status" -ForegroundColor Green
        Write-Host ""
        Write-Host "[OK] Endpoint is ready to accept requests!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Endpoint ARN: $($endpoint.EndpointArn)" -ForegroundColor Cyan
    }
    "Failed" {
        Write-Host "$status" -ForegroundColor Red
        Write-Host ""
        Write-Host "[ERROR] Endpoint creation failed!" -ForegroundColor Red
        Write-Host "Check CloudWatch logs for details." -ForegroundColor Yellow
    }
    "Updating" {
        Write-Host "$status" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The endpoint is being updated." -ForegroundColor Gray
    }
    "RollingBack" {
        Write-Host "$status" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The endpoint is rolling back to a previous version." -ForegroundColor Gray
    }
    default {
        Write-Host "$status" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Creation Time: $creationTime" -ForegroundColor DarkGray
Write-Host "Last Modified: $lastModified" -ForegroundColor DarkGray

# Check for failure reason if status is Failed
if ($status -eq "Failed") {
    $failureReason = $endpoint.FailureReason
    if ($failureReason) {
        Write-Host ""
        Write-Host "Failure Reason: $failureReason" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "To check again, run:" -ForegroundColor Cyan
Write-Host "  .\scripts\check_sagemaker_endpoint.ps1" -ForegroundColor White

