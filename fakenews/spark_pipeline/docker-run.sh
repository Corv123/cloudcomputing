#!/bin/bash
# Docker run script for Spark ML Pipeline

echo "=========================================="
echo "Spark ML Pipeline - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t fakenews-spark-pipeline:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✓ Docker image built successfully"
echo ""

# Run container
echo "Running Spark ML Pipeline..."
echo ""

docker run --rm \
    -v "$(pwd)/../datasets:/workspace/datasets:ro" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd):/workspace/spark_pipeline" \
    -e SPARK_DRIVER_MEMORY=4g \
    -e SPARK_EXECUTOR_MEMORY=4g \
    -e PYTHONPATH=/workspace \
    fakenews-spark-pipeline:latest \
    python3 spark_pipeline/main.py --datasets_dir datasets/ --output_dir output/

echo ""
echo "=========================================="
echo "Pipeline execution complete!"
echo "Check output/ directory for results"
echo "=========================================="

