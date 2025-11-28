# Create DynamoDB table for scraped news articles
# This is separate from fakenews-articles (which stores analyzed articles)

param(
    [string]$TableName = "fakenews-scraped-news",
    [string]$Region = "ap-southeast-2"
)

Write-Host "=== CREATING DYNAMODB TABLE FOR SCRAPED NEWS ===" -ForegroundColor Yellow
Write-Host ""

# Check if table already exists
$tableCheck = aws dynamodb describe-table --table-name $TableName --region $Region 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Table already exists: $TableName" -ForegroundColor Green
    exit 0
}

Write-Host "Creating table: $TableName" -ForegroundColor Cyan

# Create table
$createOutput = aws dynamodb create-table `
    --table-name $TableName `
    --attribute-definitions `
        AttributeName=id,AttributeType=S `
    --key-schema `
        AttributeName=id,KeyType=HASH `
    --billing-mode PAY_PER_REQUEST `
    --region $Region `
    --query 'TableDescription.[TableName,TableStatus,TableArn]' `
    --output text 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Table created successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "Table details:" -ForegroundColor Cyan
    $createOutput | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    Write-Host ""
    Write-Host "Waiting for table to be active..." -ForegroundColor Yellow
    aws dynamodb wait table-exists --table-name $TableName --region $Region
    Write-Host "[OK] Table is now active" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to create table: $createOutput" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== TABLE CREATED ===" -ForegroundColor Green
Write-Host ""
Write-Host "Table name: $TableName" -ForegroundColor Cyan
Write-Host "Primary key: id (String)" -ForegroundColor White
Write-Host "Billing mode: PAY_PER_REQUEST" -ForegroundColor White
Write-Host ""
Write-Host "Schema:" -ForegroundColor Yellow
Write-Host "  - id (String, HASH key)" -ForegroundColor White
Write-Host "  - title (String)" -ForegroundColor White
Write-Host "  - content (String)" -ForegroundColor White
Write-Host "  - source_url (String)" -ForegroundColor White
Write-Host "  - domain (String)" -ForegroundColor White
Write-Host "  - published_at (String)" -ForegroundColor White
Write-Host "  - ingested_at (String)" -ForegroundColor White
Write-Host "  - article_type (String) - Always 'scraped_news'" -ForegroundColor White
Write-Host "  - word_count (Number)" -ForegroundColor White
Write-Host ""

