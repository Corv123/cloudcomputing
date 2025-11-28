# Create IAM role for News Ingestion Pipeline Lambda

param(
    [string]$RoleName = "news-ingestion-lambda-role",
    [string]$Region = "ap-southeast-2"
)

Write-Host "=== CREATING IAM ROLE FOR NEWS INGESTION PIPELINE ===" -ForegroundColor Yellow
Write-Host ""

# Step 1: Create trust policy
Write-Host "Step 1: Creating trust policy..." -ForegroundColor Cyan
$trustPolicyJson = @'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
'@

$trustPolicyFile = New-TemporaryFile
$trustPolicyJson | Out-File -FilePath $trustPolicyFile.FullName -Encoding utf8 -NoNewline

try {
    $role = aws iam create-role `
        --role-name $RoleName `
        --assume-role-policy-document "file://$($trustPolicyFile.FullName)" `
        --query 'Role.Arn' `
        --output text 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Role created: $role" -ForegroundColor Green
    } else {
        if ($role -match "already exists") {
            Write-Host "[OK] Role already exists" -ForegroundColor Yellow
            $role = aws iam get-role --role-name $RoleName --query 'Role.Arn' --output text
        } else {
            Write-Host "[ERROR] Failed to create role: $role" -ForegroundColor Red
            exit 1
        }
    }
} finally {
    Remove-Item $trustPolicyFile.FullName -ErrorAction SilentlyContinue
}

Write-Host ""

# Step 2: Attach basic Lambda execution policy
Write-Host "Step 2: Attaching basic Lambda execution policy..." -ForegroundColor Cyan
aws iam attach-role-policy `
    --role-name $RoleName `
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Basic execution policy attached" -ForegroundColor Green
} else {
    Write-Host "[WARN] Could not attach basic execution policy" -ForegroundColor Yellow
}

Write-Host ""

# Step 3: Create and attach custom policy for DynamoDB and S3
Write-Host "Step 3: Creating custom policy for DynamoDB and S3..." -ForegroundColor Cyan

$policyJson = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "dynamodb:Scan",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:${Region}:*:table/fakenews-articles",
        "arn:aws:dynamodb:${Region}:*:table/fakenews-scraped-news"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fakenews-news-database/*",
        "arn:aws:s3:::fakenews-news-database"
      ]
    }
  ]
}
"@

$policyFile = New-TemporaryFile
$policyJson | Out-File -FilePath $policyFile.FullName -Encoding utf8 -NoNewline

$policyArn = aws iam create-policy `
    --policy-name news-ingestion-pipeline-policy `
    --policy-document "file://$($policyFile.FullName)" `
    --query 'Policy.Arn' `
    --output text 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Policy created: $policyArn" -ForegroundColor Green
    
    # Attach policy to role
    aws iam attach-role-policy `
        --role-name $RoleName `
        --policy-arn $policyArn 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Policy attached to role" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Could not attach policy to role" -ForegroundColor Yellow
    }
} else {
    if ($policyArn -match "already exists") {
        Write-Host "[OK] Policy already exists" -ForegroundColor Yellow
        $policyArn = aws iam get-policy --policy-arn "arn:aws:iam::979207815314:policy/news-ingestion-pipeline-policy" --query 'Policy.Arn' --output text
        aws iam attach-role-policy --role-name $RoleName --policy-arn $policyArn 2>&1 | Out-Null
    } else {
        Write-Host "[WARN] Could not create policy: $policyArn" -ForegroundColor Yellow
    }
}

Remove-Item $policyFile.FullName -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== IAM ROLE SETUP COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Role ARN: $role" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now deploy the Lambda function with:" -ForegroundColor Yellow
Write-Host "  .\deploy_lambda.ps1 -RoleName $RoleName" -ForegroundColor White
Write-Host ""

