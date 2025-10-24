FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
	python - <<'PY'
from nltk import download
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
	try:
		download(pkg)
	except Exception:
		pass
PY

COPY src ./src
COPY api ./api
COPY configs ./configs
COPY models ./models

ENV MODEL_PATH=models/sensationalism_model.joblib
ENV VECTORIZER_PATH=models/tfidf_vectorizer.joblib

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"] 