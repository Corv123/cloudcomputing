# Deploy SageMaker Endpoint for Fake News Detector ML Model
# This script packages the model and inference code, uploads to S3, and creates a SageMaker endpoint

param(
    [string]$ModelName = "fakenews-sensationalism-model",
    [string]$EndpointName = "fakenews-sensationalism-endpoint",
    [string]$InstanceType = "ml.t2.medium",
    [string]$Region = "ap-southeast-2",
    [string]$S3Bucket = "",
    [string]$RoleArn = ""
)

Write-Host "=== SageMaker Endpoint Deployment ===" -ForegroundColor Green
Write-Host ""

# Check if AWS CLI is available
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: AWS CLI not found. Please install AWS CLI first." -ForegroundColor Red
    exit 1
}

# Get S3 bucket if not provided
if ([string]::IsNullOrEmpty($S3Bucket)) {
    Write-Host "S3 bucket not specified. Please provide S3 bucket name:" -ForegroundColor Yellow
    $S3Bucket = Read-Host
}

# Get IAM role ARN if not provided
if ([string]::IsNullOrEmpty($RoleArn)) {
    Write-Host "SageMaker execution role ARN not specified. Attempting to find/create..." -ForegroundColor Yellow
    
    # Try to get existing role
    $roleOutput = aws iam get-role --role-name SageMakerExecutionRole --region $Region 2>&1
    if ($LASTEXITCODE -eq 0) {
        try {
            $existingRole = $roleOutput | ConvertFrom-Json
            $RoleArn = $existingRole.Role.Arn
            Write-Host "Found existing role: $RoleArn" -ForegroundColor Green
        } catch {
            Write-Host "Error parsing role response, will create new role" -ForegroundColor Yellow
            $RoleArn = $null
        }
    } else {
        $RoleArn = $null
    }
    
    # Create role if it doesn't exist
    if ([string]::IsNullOrEmpty($RoleArn)) {
        Write-Host "Creating SageMaker execution role..." -ForegroundColor Yellow
        # Create trust policy JSON directly (avoiding file encoding issues)
        $trustPolicyJson = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
"@
        
        # Write JSON file without BOM
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText("$PWD\sagemaker-trust-policy.json", $trustPolicyJson, $utf8NoBom)
        
        Write-Host "  Trust policy file created" -ForegroundColor Gray
        
        # Create role
        $roleOutput = aws iam create-role `
            --role-name SageMakerExecutionRole `
            --assume-role-policy-document file://sagemaker-trust-policy.json `
            --region $Region 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            try {
                $roleResult = $roleOutput | ConvertFrom-Json
                $RoleArn = $roleResult.Role.Arn
                
                # Attach SageMaker execution policy
                aws iam attach-role-policy `
                    --role-name SageMakerExecutionRole `
                    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess `
                    --region $Region | Out-Null
                
                Write-Host "Created role: $RoleArn" -ForegroundColor Green
            } catch {
                Write-Host "ERROR: Failed to parse role creation response" -ForegroundColor Red
                Write-Host "Output: $roleOutput" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host "ERROR: Failed to create SageMaker role" -ForegroundColor Red
            Write-Host "Error output: $roleOutput" -ForegroundColor Yellow
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Model Name: $ModelName" -ForegroundColor White
Write-Host "  Endpoint Name: $EndpointName" -ForegroundColor White
Write-Host "  Instance Type: $InstanceType" -ForegroundColor White
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host "  S3 Bucket: $S3Bucket" -ForegroundColor White
Write-Host "  Role ARN: $RoleArn" -ForegroundColor White
Write-Host ""

# Step 1: Create model package directory
Write-Host "Step 1: Creating model package..." -ForegroundColor Yellow
$modelDir = "sagemaker-model-package"
if (Test-Path $modelDir) {
    Remove-Item -Path $modelDir -Recurse -Force
}
New-Item -ItemType Directory -Path "$modelDir\code" -Force | Out-Null

# Copy inference script
Copy-Item -Path "sagemaker\inference.py" -Destination "$modelDir\code\inference.py" -Force
Write-Host "  [OK] Copied inference script" -ForegroundColor Gray

# Copy features_enhanced.py (needed for feature extraction)
if (Test-Path "fakenews\src\features_enhanced.py") {
    Copy-Item -Path "fakenews\src\features_enhanced.py" -Destination "$modelDir\code\features_enhanced.py" -Force
    Write-Host "  [OK] Copied features_enhanced.py" -ForegroundColor Gray
} else {
    Write-Host "  [WARN] features_enhanced.py not found!" -ForegroundColor Red
}

# Copy model files
$modelFiles = @(
    "fakenews\models\sensationalism_model_comprehensive.joblib",
    "fakenews\models\tfidf_vectorizer_comprehensive.joblib",
    "fakenews\models\scaler_comprehensive.joblib"
)

foreach ($file in $modelFiles) {
    if (Test-Path $file) {
        Copy-Item -Path $file -Destination "$modelDir\" -Force
        Write-Host "  [OK] Copied $($file.Split('\')[-1])" -ForegroundColor Gray
    } else {
        Write-Host "  [WARN] $file not found!" -ForegroundColor Red
    }
}

# Copy requirements.txt
Copy-Item -Path "sagemaker\requirements.txt" -Destination "$modelDir\code\requirements.txt" -Force
Write-Host "  [OK] Copied requirements.txt" -ForegroundColor Gray

# Step 2: Create model.tar.gz
Write-Host ""
Write-Host "Step 2: Creating model archive..." -ForegroundColor Yellow
if (Get-Command tar -ErrorAction SilentlyContinue) {
    # Use tar if available (Windows 10+ or Git Bash)
    Set-Location $modelDir
    tar -czf ..\model.tar.gz *
    Set-Location ..
    Write-Host "  [OK] Created model.tar.gz" -ForegroundColor Gray
} else {
    # Use PowerShell compression (creates .zip, we'll rename)
    Compress-Archive -Path "$modelDir\*" -DestinationPath "model.zip" -Force
    Write-Host "  [OK] Created model.zip (note: SageMaker prefers .tar.gz)" -ForegroundColor Yellow
    Write-Host "    You may need to convert to .tar.gz manually or use Docker" -ForegroundColor Yellow
}

# Step 3: Upload to S3
Write-Host ""
Write-Host "Step 3: Uploading model to S3..." -ForegroundColor Yellow
$s3ModelPath = "s3://$S3Bucket/sagemaker-models/$ModelName/model.tar.gz"
if (Test-Path "model.tar.gz") {
    Write-Host "  Uploading model.tar.gz (this may take a few minutes)..." -ForegroundColor Gray
    $uploadOutput = aws s3 cp model.tar.gz $s3ModelPath --region $Region 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Uploaded to $s3ModelPath" -ForegroundColor Gray
    } else {
        Write-Host "  [ERROR] Failed to upload to S3" -ForegroundColor Red
        Write-Host "  Error: $uploadOutput" -ForegroundColor Yellow
        Write-Host "  Retrying upload in 5 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        $uploadOutput = aws s3 cp model.tar.gz $s3ModelPath --region $Region 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Upload failed again. Please check:" -ForegroundColor Red
            Write-Host "    1. Network connection" -ForegroundColor White
            Write-Host "    2. S3 bucket permissions" -ForegroundColor White
            Write-Host "    3. AWS credentials" -ForegroundColor White
            exit 1
        } else {
            Write-Host "  [OK] Uploaded to $s3ModelPath (on retry)" -ForegroundColor Gray
        }
    }
} elseif (Test-Path "model.zip") {
    Write-Host "  [ERROR] model.zip found instead of model.tar.gz" -ForegroundColor Red
    Write-Host "    SageMaker requires .tar.gz format" -ForegroundColor Yellow
    Write-Host "    Please install tar or use Docker to create the archive" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "  [ERROR] No model archive found!" -ForegroundColor Red
    exit 1
}

# Step 4: Create SageMaker model
Write-Host ""
Write-Host "Step 4: Creating SageMaker model..." -ForegroundColor Yellow
# Use the official AWS SageMaker scikit-learn image
# For ap-southeast-2, use the correct ECR image
# Note: The image format changed - using the newer format
$modelImage = "763104351884.dkr.ecr.$Region.amazonaws.com/sklearn-inference:1.0-1.cpu"
$modelDataUrl = $s3ModelPath

$modelConfig = @{
    ModelName = $ModelName
    PrimaryContainer = @{
        Image = $modelImage
        ModelDataUrl = $modelDataUrl
        Environment = @{}
    }
    ExecutionRoleArn = $RoleArn
} | ConvertTo-Json -Depth 10

$modelConfig | Out-File -FilePath "sagemaker-model-config.json" -Encoding UTF8

$modelOutput = aws sagemaker create-model `
    --model-name $ModelName `
    --execution-role-arn $RoleArn `
    --primary-container "Image=$modelImage,ModelDataUrl=$modelDataUrl" `
    --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Created SageMaker model: $ModelName" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Failed to create SageMaker model" -ForegroundColor Red
    Write-Host "  Error: $modelOutput" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if S3 model file exists: aws s3 ls $s3ModelPath" -ForegroundColor White
    Write-Host "  2. Verify Docker image is accessible in region $Region" -ForegroundColor White
    Write-Host "  3. Check IAM role permissions for SageMaker" -ForegroundColor White
    Write-Host "" -ForegroundColor Yellow
    Write-Host "  If Docker image error persists, the image URI may need to be updated." -ForegroundColor Yellow
    Write-Host "  You may need to use a custom container or different image." -ForegroundColor Yellow
    exit 1
}

# Step 5: Create endpoint configuration
Write-Host ""
Write-Host "Step 5: Creating endpoint configuration..." -ForegroundColor Yellow
$configName = "$EndpointName-config"

$configOutput = aws sagemaker create-endpoint-config `
    --endpoint-config-name $configName `
    --production-variants "VariantName=AllTraffic,ModelName=$ModelName,InitialInstanceCount=1,InstanceType=$InstanceType" `
    --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Created endpoint configuration: $configName" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Failed to create endpoint configuration" -ForegroundColor Red
    Write-Host "  Error: $configOutput" -ForegroundColor Yellow
    Write-Host "  Note: Model must be created successfully first" -ForegroundColor Yellow
    exit 1
}

# Step 6: Create endpoint
Write-Host ""
Write-Host "Step 6: Creating SageMaker endpoint..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Yellow

$endpointOutput = aws sagemaker create-endpoint `
    --endpoint-name $EndpointName `
    --endpoint-config-name $configName `
    --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Endpoint creation initiated: $EndpointName" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Failed to create endpoint" -ForegroundColor Red
    Write-Host "  Error: $endpointOutput" -ForegroundColor Yellow
    Write-Host "  Note: Endpoint configuration must be created successfully first" -ForegroundColor Yellow
    exit 1
}
Write-Host ""
Write-Host "Waiting for endpoint to be InService..." -ForegroundColor Yellow

# Wait for endpoint to be ready
$maxWait = 600  # 10 minutes
$elapsed = 0
do {
    Start-Sleep -Seconds 30
    $elapsed += 30
    $status = aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region --query 'EndpointStatus' --output text
    Write-Host "  Status: $status (waited $elapsed seconds)" -ForegroundColor Gray
    
    if ($status -eq "InService") {
        Write-Host ""
        Write-Host "=== Endpoint Deployed Successfully! ===" -ForegroundColor Green
        Write-Host ""
        Write-Host "Endpoint Name: $EndpointName" -ForegroundColor Cyan
        Write-Host "Endpoint ARN: $(aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region --query 'EndpointArn' --output text)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Update Lambda function to call this endpoint" -ForegroundColor White
        Write-Host "2. Grant Lambda role permission to invoke SageMaker" -ForegroundColor White
        Write-Host "3. Test the endpoint" -ForegroundColor White
        break
    }
    
    if ($elapsed -ge $maxWait) {
        Write-Host ""
        Write-Host "WARNING: Endpoint creation is taking longer than expected" -ForegroundColor Yellow
        Write-Host "Check status with: aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region" -ForegroundColor Yellow
        break
    }
} while ($true)

# Cleanup
Remove-Item -Path $modelDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "sagemaker-model-config.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "sagemaker-trust-policy.json" -Force -ErrorAction SilentlyContinue

