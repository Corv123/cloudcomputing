# Dockerfile for Lambda Container Image
# Includes ML dependencies and models (up to 10 GB limit)

# Use AWS-provided Python 3.11 base image with Lambda Runtime Interface
FROM public.ecr.aws/lambda/python:3.11

# Install ML dependencies with retry and timeout settings
# Split into smaller groups to avoid timeout issues
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    numpy==1.24.3 \
    joblib==1.3.2 \
    -t ${LAMBDA_TASK_ROOT}

# Install scipy separately (large package, prone to timeouts)
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    scipy==1.10.1 \
    -t ${LAMBDA_TASK_ROOT}

# Install scikit-learn (depends on numpy and scipy)
# Constrain to use already-installed numpy 1.24.3 (not 2.x which needs compiler)
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    --no-deps \
    scikit-learn==1.3.0 \
    -t ${LAMBDA_TASK_ROOT} && \
    /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    "numpy>=1.17.3,<2.0" \
    "scipy>=1.3.2" \
    "joblib>=0.11" \
    "threadpoolctl>=2.0.0" \
    -t ${LAMBDA_TASK_ROOT}

# Install web scraping and text processing libraries
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    requests==2.31.0 \
    beautifulsoup4==4.12.3 \
    lxml==5.1.0 \
    -t ${LAMBDA_TASK_ROOT}

# Install NLP libraries
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    nltk==3.8.1 \
    vaderSentiment==3.3.2 \
    -t ${LAMBDA_TASK_ROOT}

# Download NLTK data to /tmp (Lambda writable directory)
RUN /var/lang/bin/python3.11 -c "import nltk; nltk.data.path.append('/tmp'); nltk.download('punkt', download_dir='/tmp', quiet=True); nltk.download('vader_lexicon', download_dir='/tmp', quiet=True); nltk.download('stopwords', download_dir='/tmp', quiet=True)"

# Copy Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Copy ML models to /opt/ (Lambda layer mount point)
# Models should be in fakenews/models/ directory
COPY fakenews/models/sensationalism_model_comprehensive.joblib /opt/sensationalism_model_comprehensive.joblib
COPY fakenews/models/tfidf_vectorizer_comprehensive.joblib /opt/tfidf_vectorizer_comprehensive.joblib
COPY fakenews/models/scaler_comprehensive.joblib /opt/scaler_comprehensive.joblib

# Set the CMD to your handler (required for Lambda)
CMD [ "lambda_function.lambda_handler" ]

