# Stage 1: Build the Vite React Frontend
FROM node:20-slim AS frontend-build
WORKDIR /app
# Copy package.json and package-lock.json
COPY package*.json ./
RUN npm install
# Copy frontend source and build
COPY tsconfig*.json vite.config.ts index.html ./
COPY src/ ./src/
RUN npx vite build

# Stage 2: Build the FastAPI Backend and Package the Application
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (if any needed for MLflow/SQLite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY training/ ./training/

# Copy artifacts and pre-populated databases
COPY artifacts/ ./artifacts/
COPY cartsense_local.db ./
COPY mlflow.db ./

# Copy the built Vite frontend from Stage 1
COPY --from=frontend-build /app/dist ./dist

# Set environment variables for production
ENV DATABASE_URL="sqlite:///./cartsense_local.db"
ENV MLFLOW_TRACKING_URI="sqlite:///./mlflow.db"
ENV ARTIFACTS_DIR="./artifacts"
ENV PORT=8080

# Cloud Run sets the PORT environment variable
EXPOSE 8080

# Run the FastAPI server
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
