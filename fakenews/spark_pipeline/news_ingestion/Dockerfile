# Dockerfile for News Ingestion Pipeline Lambda
# Uses Python 3.11 Lambda base image

FROM public.ecr.aws/lambda/python:3.11

# Install dependencies (no pyspark - Lambda doesn't support it)
RUN /var/lang/bin/python3.11 -m pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=5 \
    requests==2.31.0 \
    beautifulsoup4==4.12.3 \
    lxml==5.1.0 \
    pandas==2.1.4 \
    boto3==1.34.0 \
    feedparser==6.0.10 \
    -t ${LAMBDA_TASK_ROOT}

# Copy pipeline code (paths relative to project root)
COPY fakenews/spark_pipeline/news_ingestion/lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY fakenews/spark_pipeline/news_ingestion/01_scrape_news.py ${LAMBDA_TASK_ROOT}/scrape_news.py
COPY fakenews/spark_pipeline/news_ingestion/02_clean_and_deduplicate.py ${LAMBDA_TASK_ROOT}/clean_and_deduplicate.py
COPY fakenews/spark_pipeline/news_ingestion/03_store_to_database.py ${LAMBDA_TASK_ROOT}/store_to_database.py

# Copy news_scraper from analyzers
COPY fakenews/analyzers/news_scraper.py ${LAMBDA_TASK_ROOT}/news_scraper.py

# Set handler
CMD [ "lambda_handler.lambda_handler" ]

