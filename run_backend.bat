@echo off
echo ============================================================
echo  CartSense - Local Development Startup
echo ============================================================

REM Set local environment variables (override Docker settings)
set DATABASE_URL=sqlite:///./cartsense_local.db
set MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
set MLFLOW_BACKEND_STORE_URI=sqlite:///./mlflow.db
set MLFLOW_ARTIFACT_ROOT=./mlflow_artifacts
set ARTIFACTS_DIR=./artifacts
set BACKEND_URL=http://localhost:8000
set API_HOST=0.0.0.0
set API_PORT=8000
set ALPHA=0.65
set BETA=0.20
set TOP_K=5
set DRIFT_THRESHOLD=0.5
set GIT_PYTHON_REFRESH=quiet

echo [1/3] Setting up local database...
.venv\Scripts\python.exe setup_local_db.py
if errorlevel 1 (
    echo ERROR: Database setup failed!
    pause
    exit /b 1
)

echo [2/3] Training local models (if needed)...
if not exist artifacts\tfidf_vectorizer.joblib (
    echo Training models...
    .venv\Scripts\python.exe train_local.py
    if errorlevel 1 (
        echo ERROR: Training failed!
        pause
        exit /b 1
    )
) else (
    echo Models already trained, skipping...
)

echo [3/3] Starting FastAPI backend on port 8000...
echo.
echo Backend will be available at: http://localhost:8000
echo API Docs:                     http://localhost:8000/docs
echo.
.venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000 --reload
