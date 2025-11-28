# PowerShell script for running Spark ML Pipeline in Docker

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Spark ML Pipeline - Docker Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
$dockerCheck = Get-Command docker -ErrorAction SilentlyContinue
if ($null -eq $dockerCheck) {
    Write-Host "[ERROR] Docker is not installed!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
} else {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker is installed: $dockerVersion" -ForegroundColor Green
}

Write-Host ""

# Get current directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$datasetsDir = Join-Path $projectRoot "datasets"
$outputDir = Join-Path $scriptDir "output"

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t fakenews-spark-pipeline:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Docker image built successfully" -ForegroundColor Green
Write-Host ""

# Run container
Write-Host "Running Spark ML Pipeline..." -ForegroundColor Yellow
Write-Host ""

docker run --rm `
    -v "${datasetsDir}:/workspace/datasets:ro" `
    -v "${outputDir}:/workspace/output" `
    -v "${scriptDir}:/workspace/spark_pipeline" `
    -e SPARK_DRIVER_MEMORY=4g `
    -e SPARK_EXECUTOR_MEMORY=4g `
    -e PYTHONPATH=/workspace `
    fakenews-spark-pipeline:latest `
    python3 spark_pipeline/main.py --datasets_dir datasets/ --output_dir output/

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Pipeline execution failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Pipeline execution complete!" -ForegroundColor Green
Write-Host "Check output/ directory for results" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
