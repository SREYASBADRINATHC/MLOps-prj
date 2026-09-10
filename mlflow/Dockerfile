FROM python:3.11-slim-bookworm

WORKDIR /mlflow

# Install the tracking server once at image build time.  Installing packages in
# the container command made every restart slow and left the health check stuck
# while pip was downloading dependencies.
RUN pip install --no-cache-dir mlflow==2.16.2 psycopg2-binary==2.9.9

EXPOSE 5000
