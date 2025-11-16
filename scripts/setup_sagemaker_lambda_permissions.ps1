# Grant Lambda function permission to invoke SageMaker endpoint

param(
    [string]$LambdaFunctionName = "fakenews-analyzer",
    [string]$SageMakerEndpointName = "fakenews-sensationalism-endpoint",
    [string]$Region = "ap-southeast-2"
)

Write-Host "=== Setting up SageMaker Lambda Permissions ===" -ForegroundColor Green
Write-Host ""

# Get Lambda function role
Write-Host "Getting Lambda function role..." -ForegroundColor Yellow
$lambdaConfig = aws lambda get-function-configuration `
    --function-name $LambdaFunctionName `
    --region $Region `
    | ConvertFrom-Json

if (-not $lambdaConfig) {
    Write-Host "ERROR: Lambda function not found: $LambdaFunctionName" -ForegroundColor Red
    exit 1
}

$roleArn = $lambdaConfig.Role
$roleName = $roleArn.Split('/')[-1]

Write-Host "Lambda Role: $roleName" -ForegroundColor Cyan
Write-Host "Role ARN: $roleArn" -ForegroundColor Cyan
Write-Host ""

# Create policy document for SageMaker invoke
$accountId = (aws sts get-caller-identity --query Account --output text)
$resourceArn = "arn:aws:sagemaker:${Region}:${accountId}:endpoint/${SageMakerEndpointName}"

$policyDoc = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "$resourceArn"
    }
  ]
}
"@

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("$PWD\sagemaker-invoke-policy.json", $policyDoc, $utf8NoBom)

# Attach policy to Lambda role
Write-Host "Attaching SageMaker invoke policy to Lambda role..." -ForegroundColor Yellow
aws iam put-role-policy `
    --role-name $roleName `
    --policy-name SageMakerInvokePolicy `
    --policy-document file://sagemaker-invoke-policy.json `
    --region $Region

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Policy attached successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to attach policy" -ForegroundColor Red
    exit 1
}

# Set Lambda environment variable for SageMaker endpoint
Write-Host ""
Write-Host "Setting Lambda environment variable..." -ForegroundColor Yellow
$currentEnv = aws lambda get-function-configuration `
    --function-name $LambdaFunctionName `
    --region $Region `
    --query 'Environment.Variables' `
    | ConvertFrom-Json

if (-not $currentEnv) {
    $currentEnv = @{}
}

$currentEnv.SAGEMAKER_ENDPOINT_NAME = $SageMakerEndpointName

$envJson = @{
    Variables = $currentEnv
} | ConvertTo-Json -Depth 10

$envJson | Out-File -FilePath "lambda-env-sagemaker.json" -Encoding UTF8

aws lambda update-function-configuration `
    --function-name $LambdaFunctionName `
    --environment "Variables={$(($currentEnv.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ',')},SAGEMAKER_ENDPOINT_NAME=$SageMakerEndpointName" `
    --region $Region | Out-Null

# Better approach - use JSON file
$updateConfig = @{
    FunctionName = $LambdaFunctionName
    Environment = @{
        Variables = $currentEnv
    }
} | ConvertTo-Json -Depth 10

$updateConfig | Out-File -FilePath "lambda-update-sagemaker.json" -Encoding UTF8

aws lambda update-function-configuration `
    --cli-input-json file://lambda-update-sagemaker.json `
    --region $Region | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Environment variable set successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to set environment variable" -ForegroundColor Yellow
    Write-Host "  You may need to set it manually:" -ForegroundColor Yellow
    Write-Host "  SAGEMAKER_ENDPOINT_NAME=$SageMakerEndpointName" -ForegroundColor White
}

# Cleanup
Remove-Item -Path "sagemaker-invoke-policy.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "lambda-env-sagemaker.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "lambda-update-sagemaker.json" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Lambda function can now invoke SageMaker endpoint:" -ForegroundColor Cyan
Write-Host "  Endpoint: $SageMakerEndpointName" -ForegroundColor White
Write-Host "  Environment Variable: SAGEMAKER_ENDPOINT_NAME" -ForegroundColor White

