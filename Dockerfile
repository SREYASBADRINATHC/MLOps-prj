# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install system dependencies, including OpenJDK 17 (required for PySpark JVM execution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    procps \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy the server script and any utility scripts
COPY cartsense_api.py /app/cartsense_api.py
COPY data_ingestion.py /app/data_ingestion.py
COPY retrain_trigger.py /app/retrain_trigger.py

# Install required Python dependencies
# We install pandas, scikit-learn, pyspark, mlflow, psycopg2-binary, sqlalchemy, fastapi, uvicorn, and evidently
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    pyspark==3.5.1 \
    mlflow==2.10.2 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.28 \
    evidently==0.4.16

# Expose port 8000 for FastAPI Server
EXPOSE 8000

# Start FastAPI application using uvicorn
CMD ["uvicorn", "cartsense_api:app", "--host", "0.0.0.0", "--port", "8000"]
