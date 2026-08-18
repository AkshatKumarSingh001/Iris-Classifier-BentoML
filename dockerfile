FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENTOML_HOME=/app/.bentoml

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model and save it to BentoML's model store before serving
RUN python train.py

EXPOSE 3000

CMD ["bash", "-lc", "bentoml serve service.py:IrisClassifier --host 0.0.0.0 --port 3000"]

