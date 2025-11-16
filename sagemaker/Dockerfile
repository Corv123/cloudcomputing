# Dockerfile for SageMaker scikit-learn inference container
# Using python:3.11-slim-bullseye (older tag, more likely Docker v2 format)
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml

# Copy requirements and install Python dependencies
COPY code/requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copy inference code
COPY code/ /opt/ml/code/

# Create a startup script that loads models then starts gunicorn
RUN echo '#!/bin/bash\n\
cd /opt/ml/code\n\
echo "[INFO] Loading models..."\n\
python -c "import inference; inference.model_fn(\"/opt/ml/model\")"\n\
echo "[INFO] Models loaded, starting gunicorn..."\n\
exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 60 inference:app\n\
' > /opt/ml/start.sh && chmod +x /opt/ml/start.sh

# Set Python path
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# SageMaker expects the model to be in /opt/ml/model
# The model files will be extracted here from the model.tar.gz

# Set the entrypoint to use our startup script
ENTRYPOINT ["/opt/ml/start.sh"]

