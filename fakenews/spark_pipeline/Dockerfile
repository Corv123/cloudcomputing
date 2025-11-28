# Dockerfile for Spark ML Pipeline
# Based on Apache Spark PySpark image

FROM apache/spark-py:latest

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/
# Use --user flag to install to user directory, or run as root
USER root
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to avoid runtime errors
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || true

# Copy pipeline code (includes features_enhanced.py if copied to this directory)
COPY . /workspace/spark_pipeline/

# Note: Datasets should be mounted as volume or in S3
# For local testing, mount datasets directory when running container

# Set Python path to include Spark's Python libraries
ENV PYTHONPATH=/workspace:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip

# Default command
CMD ["bash"]

