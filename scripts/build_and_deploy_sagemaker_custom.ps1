# Build custom Docker container and deploy SageMaker endpoint

param(
    [string]$ModelName = "fakenews-sensationalism-model",
    [string]$EndpointName = "fakenews-sensationalism-endpoint",
    [string]$InstanceType = "ml.t2.medium",
    [string]$Region = "ap-southeast-2",
    [string]$S3Bucket = "",
    [string]$RoleArn = "",
    [string]$ECRRepository = "fakenews-sagemaker-inference"
)

Write-Host "=== Custom SageMaker Container Deployment ===" -ForegroundColor Green
Write-Host ""

# Check prerequisites
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: AWS CLI not found" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Docker not found. Please install Docker Desktop." -ForegroundColor Red
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
    
    $roleOutput = aws iam get-role --role-name SageMakerExecutionRole --region $Region 2>&1
    if ($LASTEXITCODE -eq 0) {
        try {
            $existingRole = $roleOutput | ConvertFrom-Json
            $RoleArn = $existingRole.Role.Arn
            Write-Host "Found existing role: $RoleArn" -ForegroundColor Green
        } catch {
            $RoleArn = $null
        }
    } else {
        $RoleArn = $null
    }
    
    if ([string]::IsNullOrEmpty($RoleArn)) {
        Write-Host "Creating SageMaker execution role..." -ForegroundColor Yellow
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
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText("$PWD\sagemaker-trust-policy.json", $trustPolicyJson, $utf8NoBom)
        
        $roleOutput = aws iam create-role `
            --role-name SageMakerExecutionRole `
            --assume-role-policy-document file://sagemaker-trust-policy.json `
            --region $Region 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            try {
                $roleResult = $roleOutput | ConvertFrom-Json
                $RoleArn = $roleResult.Role.Arn
                
                aws iam attach-role-policy `
                    --role-name SageMakerExecutionRole `
                    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess `
                    --region $Region | Out-Null
                
                Write-Host "Created role: $RoleArn" -ForegroundColor Green
            } catch {
                Write-Host "ERROR: Failed to create SageMaker role" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "ERROR: Failed to create SageMaker role" -ForegroundColor Red
            Write-Host "Error: $roleOutput" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Get AWS account ID
$accountId = aws sts get-caller-identity --query Account --output text --region $Region
if ([string]::IsNullOrEmpty($accountId)) {
    Write-Host "ERROR: Could not get AWS account ID" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Model Name: $ModelName" -ForegroundColor White
Write-Host "  Endpoint Name: $EndpointName" -ForegroundColor White
Write-Host "  Instance Type: $InstanceType" -ForegroundColor White
Write-Host "  Region: $Region" -ForegroundColor White
Write-Host "  S3 Bucket: $S3Bucket" -ForegroundColor White
Write-Host "  Role ARN: $RoleArn" -ForegroundColor White
Write-Host "  AWS Account: $accountId" -ForegroundColor White
Write-Host "  ECR Repository: $ECRRepository" -ForegroundColor White
Write-Host ""

# Step 1: Package model files
Write-Host "Step 1: Packaging model files..." -ForegroundColor Yellow
$modelDir = "sagemaker-model-package"
if (Test-Path $modelDir) {
    Remove-Item -Path $modelDir -Recurse -Force
}
New-Item -ItemType Directory -Path "$modelDir\code" -Force | Out-Null

# Copy inference code
Copy-Item -Path "sagemaker\inference.py" -Destination "$modelDir\code\inference.py" -Force
Copy-Item -Path "fakenews\src\features_enhanced.py" -Destination "$modelDir\code\features_enhanced.py" -Force
Copy-Item -Path "sagemaker\requirements.txt" -Destination "$modelDir\code\requirements.txt" -Force

# Copy model files
Copy-Item -Path "fakenews\models\sensationalism_model_comprehensive.joblib" -Destination "$modelDir\" -Force
Copy-Item -Path "fakenews\models\tfidf_vectorizer_comprehensive.joblib" -Destination "$modelDir\" -Force
Copy-Item -Path "fakenews\models\scaler_comprehensive.joblib" -Destination "$modelDir\" -Force

Write-Host "  [OK] Model files packaged" -ForegroundColor Gray

# Step 2: Create model.tar.gz
Write-Host ""
Write-Host "Step 2: Creating model archive..." -ForegroundColor Yellow
if (Get-Command tar -ErrorAction SilentlyContinue) {
    Set-Location $modelDir
    tar -czf ..\model.tar.gz *
    Set-Location ..
    Write-Host "  [OK] Created model.tar.gz" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] tar not found. Please install tar or Git Bash." -ForegroundColor Red
    exit 1
}

# Step 3: Upload to S3
Write-Host ""
Write-Host "Step 3: Uploading model to S3..." -ForegroundColor Yellow
$s3ModelPath = "s3://$S3Bucket/sagemaker-models/$ModelName/model.tar.gz"
Write-Host "  Uploading model.tar.gz..." -ForegroundColor Gray
$uploadOutput = aws s3 cp model.tar.gz $s3ModelPath --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Uploaded to $s3ModelPath" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Failed to upload to S3" -ForegroundColor Red
    Write-Host "  Error: $uploadOutput" -ForegroundColor Yellow
    exit 1
}

# Step 4: Create ECR repository
Write-Host ""
Write-Host "Step 4: Setting up ECR repository..." -ForegroundColor Yellow
$ecrUri = "$accountId.dkr.ecr.$Region.amazonaws.com/$ECRRepository"
$ecrCheck = aws ecr describe-repositories --repository-names $ECRRepository --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating ECR repository..." -ForegroundColor Gray
    aws ecr create-repository --repository-name $ECRRepository --region $Region | Out-Null
    Write-Host "  [OK] ECR repository created" -ForegroundColor Gray
} else {
    Write-Host "  [OK] ECR repository already exists" -ForegroundColor Gray
}

# Step 5: Build Docker image
Write-Host ""
Write-Host "Step 5: Building Docker image..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Gray

# Copy Dockerfile to model directory
Copy-Item -Path "sagemaker\Dockerfile" -Destination "$modelDir\Dockerfile" -Force

# Build Docker image - change to directory first (simpler approach)
$originalDir = Get-Location
Push-Location $modelDir

Write-Host "  Building Docker image from current directory..." -ForegroundColor Gray

# Build and push directly to ECR to preserve Docker v2 format
# SageMaker requires Docker v2 Schema 2, not OCI format
# Pushing directly during build helps preserve the format
Write-Host "  Building and pushing directly to ECR (preserves Docker v2 format)..." -ForegroundColor DarkGray

# Get ECR login first (needed for direct push)
$ecrLoginOutput = aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ecrUri 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] ECR login failed, will build locally first..." -ForegroundColor Yellow
    $ecrLoginOutput = $null
}

$localImageTag = "${ECRRepository}:latest"
$targetImageTag = "${ecrUri}:latest"

# Try building and pushing directly to ECR (preserves format better)
if ($ecrLoginOutput -ne $null) {
    Write-Host "  Building and pushing to ECR (no cache to ensure latest requirements)..." -ForegroundColor DarkGray
    $buildOutput = docker buildx build `
        --platform linux/amd64 `
        --tag "$targetImageTag" `
        --output type=registry `
        --provenance=false `
        --sbom=false `
        --no-cache `
        . 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        # Also tag locally for reference
        docker pull "$targetImageTag" 2>&1 | Out-Null
        docker tag "$targetImageTag" "$localImageTag" 2>&1 | Out-Null
        Write-Host "  [OK] Built and pushed directly to ECR" -ForegroundColor DarkGray
        $buildSuccess = $true
    } else {
        $buildSuccess = $false
    }
} else {
    $buildSuccess = $false
}

# Fallback: build locally then push
if (-not $buildSuccess) {
    Write-Host "  Building locally (fallback, no cache)..." -ForegroundColor DarkGray
    $env:DOCKER_BUILDKIT = "0"
    $buildOutput = docker build -t "$localImageTag" --platform linux/amd64 --no-cache . 2>&1
    $env:DOCKER_BUILDKIT = $null
    $buildSuccess = ($LASTEXITCODE -eq 0)
}

# Display output
$buildOutput | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "Step \d+/\d+" -or $line -match "ERROR" -or $line -match "Successfully" -or $line -match "=>" -or $line -match "Building" -or $line -match "Pulling" -or $line -match "CACHED") {
        Write-Host "  $line" -ForegroundColor Gray
    }
}

# Return to original directory
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Docker build failed" -ForegroundColor Red
    Write-Host "  Make sure Docker Desktop is running" -ForegroundColor Yellow
    Write-Host "  Try: docker buildx ls (to check buildx is available)" -ForegroundColor Yellow
    exit 1
}
if ($buildSuccess) {
    Write-Host "  [OK] Docker image built" -ForegroundColor Gray
} else {
    Write-Host "  [ERROR] Docker build failed" -ForegroundColor Red
    Write-Host "  Make sure Docker Desktop is running" -ForegroundColor Yellow
    exit 1
}

# Step 6: Push to ECR (skip if already pushed during build)
Write-Host ""
Write-Host "Step 6: Pushing image to ECR..." -ForegroundColor Yellow

if ($buildSuccess -and $ecrLoginOutput -ne $null) {
    Write-Host "  [OK] Image already pushed to ECR during build" -ForegroundColor Gray
    $sourceImage = $localImageTag
    $targetImage = $targetImageTag
} else {
    # Get ECR login
    Write-Host "  Logging into ECR..." -ForegroundColor Gray
    $loginOutput = aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ecrUri 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] Failed to login to ECR" -ForegroundColor Red
        Write-Host "  Error: $loginOutput" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  [OK] Logged into ECR" -ForegroundColor Gray

    # Delete existing image with manifest index issue (if it exists)
    Write-Host "  Removing existing image from ECR (to fix manifest issue)..." -ForegroundColor Gray
    aws ecr batch-delete-image `
        --repository-name $ECRRepository `
        --image-ids imageTag=latest `
        --region $Region 2>&1 | Out-Null
    # Ignore errors if image doesn't exist
    Write-Host "  [OK] Cleaned up existing image" -ForegroundColor DarkGray

    # Tag image
    Write-Host "  Tagging image..." -ForegroundColor Gray
    $sourceImage = "${ECRRepository}:latest"
    $targetImage = "${ecrUri}:latest"
    Write-Host "  Source: $sourceImage" -ForegroundColor DarkGray
    Write-Host "  Target: $targetImage" -ForegroundColor DarkGray
    docker tag "$sourceImage" "$targetImage" 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [ERROR] Failed to tag image" -ForegroundColor Red
        Write-Host "  Source: $sourceImage" -ForegroundColor Yellow
        Write-Host "  Target: $targetImage" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  [OK] Image tagged" -ForegroundColor Gray
}

# Push image with retry logic (skip if already pushed during build)
if ($buildSuccess -and $ecrLoginOutput -ne $null) {
    Write-Host "  [OK] Image already in ECR, skipping push" -ForegroundColor Gray
    $pushSuccess = $true
} else {
    Write-Host "  Pushing image (this may take a few minutes)..." -ForegroundColor Gray
    $maxRetries = 3
    $retryCount = 0
    $pushSuccess = $false

    while ($retryCount -lt $maxRetries -and -not $pushSuccess) {
        if ($retryCount -gt 0) {
            Write-Host "  Retry attempt $retryCount of $maxRetries..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
        
        $pushOutput = docker push "$targetImage" 2>&1
        
        # Display push progress
        $pushOutput | ForEach-Object {
            $line = $_.ToString()
            if ($line -match "The push refers to" -or $line -match "Pushed" -or $line -match "latest: digest" -or $line -match "Layer" -or $line -match "digest:") {
                Write-Host "  $line" -ForegroundColor Gray
            }
        }
        
        if ($LASTEXITCODE -eq 0) {
            $pushSuccess = $true
            Write-Host "  [OK] Image pushed to ECR" -ForegroundColor Gray
        } else {
            $retryCount++
            if ($retryCount -lt $maxRetries) {
                Write-Host "  [WARN] Push failed, will retry..." -ForegroundColor Yellow
                # Check if it's a network error (EOF, timeout, etc.)
                $errorStr = $pushOutput -join " "
                if ($errorStr -match "EOF" -or $errorStr -match "timeout" -or $errorStr -match "connection") {
                    Write-Host "  Network error detected, waiting longer before retry..." -ForegroundColor Yellow
                    Start-Sleep -Seconds 10
                }
            } else {
                Write-Host "  [ERROR] Failed to push image to ECR after $maxRetries attempts" -ForegroundColor Red
                Write-Host "  Last error output:" -ForegroundColor Yellow
                $pushOutput | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
                Write-Host ""
                Write-Host "  You can try pushing manually with:" -ForegroundColor Cyan
                Write-Host "    docker push `"$targetImage`"" -ForegroundColor White
                exit 1
            }
        }
    }
}

# Step 7: Clean up existing resources (if any)
Write-Host ""
Write-Host "Step 7: Cleaning up existing resources (if any)..." -ForegroundColor Yellow

# Check and delete endpoint if exists
$endpointCheck = aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    $endpointStatus = ($endpointCheck | ConvertFrom-Json).EndpointStatus
    if ($endpointStatus -ne "Deleting") {
        Write-Host "  Deleting existing endpoint: $EndpointName" -ForegroundColor Gray
        aws sagemaker delete-endpoint --endpoint-name $EndpointName --region $Region 2>&1 | Out-Null
        Write-Host "  [OK] Endpoint deletion initiated" -ForegroundColor DarkGray
    }
}

# Check and delete endpoint config if exists
$configName = "${EndpointName}-config"
$configCheck = aws sagemaker describe-endpoint-config --endpoint-config-name $configName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Deleting existing endpoint config: $configName" -ForegroundColor Gray
    aws sagemaker delete-endpoint-config --endpoint-config-name $configName --region $Region 2>&1 | Out-Null
    Write-Host "  [OK] Endpoint config deleted" -ForegroundColor DarkGray
}

# Check and delete model if exists
$modelCheck = aws sagemaker describe-model --model-name $ModelName --region $Region 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Deleting existing model: $ModelName" -ForegroundColor Gray
    aws sagemaker delete-model --model-name $ModelName --region $Region 2>&1 | Out-Null
    Write-Host "  [OK] Model deleted" -ForegroundColor DarkGray
}

# Wait a bit for deletions to complete
if ($LASTEXITCODE -eq 0 -or $configCheck -or $modelCheck) {
    Write-Host "  Waiting 5 seconds for deletions to complete..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
}

# Step 8: Create SageMaker model
Write-Host ""
Write-Host "Step 8: Creating SageMaker model..." -ForegroundColor Yellow

# Verify variables are set
if ([string]::IsNullOrEmpty($ecrUri)) {
    Write-Host "  [ERROR] ECR URI is empty!" -ForegroundColor Red
    exit 1
}
if ([string]::IsNullOrEmpty($s3ModelPath)) {
    Write-Host "  [ERROR] S3 model path is empty!" -ForegroundColor Red
    exit 1
}

$modelImage = "${ecrUri}:latest"
Write-Host "  Model Image: $modelImage" -ForegroundColor DarkGray
Write-Host "  Model Data: $s3ModelPath" -ForegroundColor DarkGray

# Create model configuration JSON - use explicit string values
$modelConfigJson = @"
{
  "ModelName": "$ModelName",
  "ExecutionRoleArn": "$RoleArn",
  "PrimaryContainer": {
    "Image": "$modelImage",
    "ModelDataUrl": "$s3ModelPath"
  }
}
"@

$modelConfigFile = "sagemaker-model-config.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("$PWD\$modelConfigFile", $modelConfigJson, $utf8NoBom)

# Verify the JSON file was created correctly
if (Test-Path $modelConfigFile) {
    $jsonContent = Get-Content $modelConfigFile -Raw
    Write-Host "  Generated JSON:" -ForegroundColor DarkGray
    Write-Host $jsonContent -ForegroundColor DarkGray
} else {
    Write-Host "  [ERROR] Failed to create JSON config file" -ForegroundColor Red
    exit 1
}

$modelOutput = aws sagemaker create-model `
    --cli-input-json file://$modelConfigFile `
    --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Created SageMaker model: $ModelName" -ForegroundColor Gray
    Remove-Item -Path $modelConfigFile -ErrorAction SilentlyContinue
} else {
    Write-Host "  [ERROR] Failed to create SageMaker model" -ForegroundColor Red
    Write-Host "  Error: $modelOutput" -ForegroundColor Yellow
    Write-Host "  Config file: $modelConfigFile" -ForegroundColor Yellow
    exit 1
}

# Step 9: Create endpoint configuration
Write-Host ""
Write-Host "Step 9: Creating endpoint configuration..." -ForegroundColor Yellow
$configName = "${EndpointName}-config"

# Create endpoint config JSON
$configJson = @{
    EndpointConfigName = $configName
    ProductionVariants = @(
        @{
            VariantName = "AllTraffic"
            ModelName = $ModelName
            InitialInstanceCount = 1
            InstanceType = $InstanceType
        }
    )
} | ConvertTo-Json -Depth 10

$configFile = "sagemaker-endpoint-config.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("$PWD\$configFile", $configJson, $utf8NoBom)

$configOutput = aws sagemaker create-endpoint-config `
    --cli-input-json file://$configFile `
    --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Created endpoint configuration: $configName" -ForegroundColor Gray
    Remove-Item -Path $configFile -ErrorAction SilentlyContinue
} else {
    Write-Host "  [ERROR] Failed to create endpoint configuration" -ForegroundColor Red
    Write-Host "  Error: $configOutput" -ForegroundColor Yellow
    Remove-Item -Path $configFile -ErrorAction SilentlyContinue
    exit 1
}

# Step 10: Create endpoint
Write-Host ""
Write-Host "Step 10: Creating SageMaker endpoint..." -ForegroundColor Yellow
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
    exit 1
}

# Step 10: Wait for endpoint
Write-Host ""
Write-Host "Waiting for endpoint to be InService..." -ForegroundColor Yellow
$maxWait = 600
$elapsed = 0
do {
    Start-Sleep -Seconds 30
    $elapsed += 30
    $statusOutput = aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region --query 'EndpointStatus' --output text 2>&1
    if ($LASTEXITCODE -eq 0) {
        $status = $statusOutput.Trim()
        Write-Host "  Status: $status (waited $elapsed seconds)" -ForegroundColor Gray
        
        if ($status -eq "InService") {
            Write-Host ""
            Write-Host "=== Endpoint Deployed Successfully! ===" -ForegroundColor Green
            Write-Host ""
            Write-Host "Endpoint Name: $EndpointName" -ForegroundColor Cyan
            Write-Host "Endpoint ARN: $(aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region --query 'EndpointArn' --output text)" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Next steps:" -ForegroundColor Yellow
            Write-Host "1. Grant Lambda permissions: .\scripts\setup_sagemaker_lambda_permissions.ps1" -ForegroundColor White
            Write-Host "2. Deploy updated Lambda function" -ForegroundColor White
            Write-Host "3. Test the endpoint" -ForegroundColor White
            break
        }
    } else {
        Write-Host "  Error checking status: $statusOutput" -ForegroundColor Yellow
    }
    
    if ($elapsed -ge $maxWait) {
        Write-Host ""
        Write-Host "WARNING: Endpoint creation is taking longer than expected" -ForegroundColor Yellow
        Write-Host "Check status: aws sagemaker describe-endpoint --endpoint-name $EndpointName --region $Region" -ForegroundColor Yellow
        break
    }
} while ($true)

# Cleanup
Remove-Item -Path $modelDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "sagemaker-trust-policy.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "model.tar.gz" -Force -ErrorAction SilentlyContinue

