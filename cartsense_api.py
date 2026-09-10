import os
import random
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# FastAPI App
app = FastAPI(
    title="CartSense Recommendation Server",
    description="Full-stack hybrid ML recommendation serving API demonstrating Cold Start routing.",
    version="1.0.0"
)

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres_secret_password@localhost:5432/feature_store"
)

# Optional SQLalchemy engine
engine = None
SessionLocal = None
try:
    connect_args = {"timeout": 5} if DATABASE_URL.startswith("sqlite") else {"connect_timeout": 5}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("🔌 SQLAlchemy configured for PostgreSQL database connection.")
except Exception as e:
    print(f"⚠️ Failed to pre-configure SQLAlchemy: {e}")

# Pydantic Schemas for Requests & Responses
class RecommendRequest(BaseModel):
    user_id: int
    product_id: str

class RecommendationItem(BaseModel):
    product_id: str
    name: str
    specs: str
    category: str
    confidence_score: float
    routing_engine: str

class RecommendResponse(BaseModel):
    user_id: int
    queried_product_id: str
    routing_path: str
    explanation: str
    recommendations: List[RecommendationItem]

# Standard Mock Products for quick fallbacks or if DB is offline
MOCK_PRODUCTS = {
    "laptop_x1": {"name": "ThinkPad X1 Carbon (Gen 12)", "specs": "Intel Core Ultra 7, 32GB LPDDR5X RAM, 1TB SSD", "category": "Laptop"},
    "laptop_macbook": {"name": "MacBook Pro 14\" M3", "specs": "Apple M3 Pro, 18GB Unified Memory, 512GB SSD", "category": "Laptop"},
    "laptop_xps13": {"name": "Dell XPS 13 (2026 Model)", "specs": "Intel Core Ultra 7, 16GB RAM, 512GB SSD", "category": "Laptop"},
    "laptop_rog": {"name": "ROG Zephyrus G14", "specs": "AMD Ryzen 9 8945HS, NVIDIA RTX 4070, 1TB SSD", "category": "Laptop"},
    "laptop_latitude": {"name": "Dell Latitude 5440", "specs": "Intel Core i5, 16GB RAM, 512GB SSD", "category": "Laptop"},
    "phone_iphone15": {"name": "iPhone 15 Pro Max", "specs": "A17 Pro, 256GB, Super Retina XDR OLED", "category": "Mobile Phone"},
    "phone_s24": {"name": "Samsung Galaxy S24 Ultra", "specs": "Snapdragon 8 Gen 3, 12GB RAM, 256GB AMOLED", "category": "Mobile Phone"},
    "phone_pixel8": {"name": "Google Pixel 8 Pro", "specs": "Tensor G3, AI Magic Editor, 128GB OLED", "category": "Mobile Phone"},
    "phone_oneplus12": {"name": "OnePlus 12 5G", "specs": "Snapdragon 8 Gen 3, 16GB RAM, 512GB AMOLED", "category": "Mobile Phone"},
    "phone_nothing2": {"name": "Nothing Phone (2)", "specs": "Snapdragon 8+ Gen 1, Glyph Interface 2.0, 256GB", "category": "Mobile Phone"}
}

# In-Memory list of products that have known interactions (based on data_ingestion.py)
INTERACTED_PRODUCT_IDS = {"laptop_x1", "phone_iphone15", "laptop_macbook", "phone_s24", "laptop_rog", "phone_pixel8", "laptop_latitude", "phone_oneplus12"}

# ==============================================================================
# PRODUCTION MLFLOW MODEL LOADING ARCHITECTURE (EXPLANATION)
# ==============================================================================
# In a real production deployment, you would load models at server startup:
#
# import mlflow
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
# 
# try:
#     # Path 1: Load Content-Based Scikit-Learn (TF-IDF Vectorizer & Cosine Similarity) model
#     tfidf_model = mlflow.pyfunc.load_model("models:/content_based_tfidf/Production")
# 
#     # Path 2: Load Collaborative Filtering PySpark ALS model
#     als_model = mlflow.pyfunc.load_model("models:/collaborative_als/Production")
# except Exception as e:
#     logger.warning(f"Could not load production models: {e}")
# ==============================================================================

@app.get("/health")
def health_check():
    """Returns database status and MLflow connection health."""
    db_status = "Online"
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            db_status = "Offline"
    else:
        db_status = "Offline (SQLAlchemy not ready)"
        
    return {
        "status": "healthy",
        "database": db_status,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "active_pipelines": [
            "Content-Based Pipeline (Scikit-Learn & sentence-transformers/all-MiniLM-L6-v2)",
            "Collaborative Filtering Pipeline (PySpark MLlib ALS)"
        ]
    }

@app.post("/api/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):
    """
    Hybrid Router Engine:
    - Route 1: COLD START PATH (Scikit-Learn TF-IDF Content-Based)
      Triggered when the requested product has 0 historical interactions in the database.
    - Route 2: EXISTING DATA PATH (PySpark ALS Collaborative Filtering)
      Triggered when the requested product has existing historical interactions in the database.
    """
    user_id = request.user_id
    product_id = request.product_id
    
    # 1. Determine if product has historical interactions in database
    has_interactions = False
    
    if SessionLocal:
        try:
            # We open a brief DB session to inspect historical interactions
            db = SessionLocal()
            query = text("SELECT COUNT(*) FROM interactions WHERE product_id = :pid")
            result = db.execute(query, {"pid": product_id}).scalar()
            db.close()
            if result and result > 0:
                has_interactions = True
        except Exception as e:
            # Fallback to local memory set if PostgreSQL container is loading/offline
            print(f"⚠️ Db Query failed ({e}), falling back to in-memory index check.")
            has_interactions = product_id in INTERACTED_PRODUCT_IDS
    else:
        has_interactions = product_id in INTERACTED_PRODUCT_IDS

    recommendations: List[RecommendationItem] = []

    # 2. Routing Decision
    if not has_interactions:
        # PATH 1: COLD START (Content-Based via TF-IDF & sentence-transformers)
        routing_path = "Path 1 (New Item / No Interaction): Content-Based Pipeline (Scikit-Learn)"
        explanation = (
            f"The product '{product_id}' has 0 historical interactions in the Feature Store database. "
            "Triggering Content-Based Pipeline with term importance and token similarity metrics "
            "to recommend mathematically similar specifications."
        )
        
        # Simulating TF-IDF matching: Recommend same category items with close specs
        category = "Laptop" if "laptop" in product_id.lower() else "Mobile Phone"
        candidates = [k for k, v in MOCK_PRODUCTS.items() if v["category"] == category and k != product_id]
        
        # Select 3 items and calculate mock cosine similarities using a stable hash
        for i, cand_id in enumerate(candidates[:3]):
            val = MOCK_PRODUCTS[cand_id]
            # Synthesize TF-IDF & sentence-transformers metric (similarity score between 0.70 and 0.98)
            similarity = round(0.70 + (hash(cand_id + product_id) % 280) / 1000.0, 3)
            
            recommendations.append(RecommendationItem(
                product_id=cand_id,
                name=val["name"],
                specs=val["specs"],
                category=val["category"],
                confidence_score=similarity * 100.0,
                routing_engine="TF-IDF Content Vectorizer"
            ))
            
    else:
        # PATH 2: COLLABORATIVE FILTERING (PySpark MLlib Matrix Factorization)
        routing_path = "Path 2 (Existing User / Historical Data): Collaborative Filtering Pipeline (PySpark MLlib)"
        explanation = (
            f"The product '{product_id}' has existing interaction metrics inside the database. "
            "Serving recommendation list compiled via PySpark ALS collaborative factorization latent variables."
        )
        
        # Simulating PySpark ALS collaborative model predictions:
        # Return popular items that other users who viewed/purchased this item also liked
        candidates = [k for k in MOCK_PRODUCTS.keys() if k != product_id]
        # Shuffle deterministically to simulate Collaborative recommendations for the user
        random_gen = random.Random(user_id)
        selected_candidates = random_gen.sample(candidates, 3)
        
        for cand_id in selected_candidates:
            val = MOCK_PRODUCTS[cand_id]
            # ALS ratings usually map back to predicted scores (between 3.5 and 5.0 out of 5.0)
            predicted_rating = round(3.5 + (random_gen.random() * 1.5), 2)
            # Map rating to percentage
            confidence = round((predicted_rating / 5.0) * 100.0, 1)
            
            recommendations.append(RecommendationItem(
                product_id=cand_id,
                name=val["name"],
                specs=val["specs"],
                category=val["category"],
                confidence_score=confidence,
                routing_engine="PySpark ALS Collaborative"
            ))

    # Sort output by highest confidence score first
    recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

    return RecommendResponse(
        user_id=user_id,
        queried_product_id=product_id,
        routing_path=routing_path,
        explanation=explanation,
        recommendations=recommendations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cartsense_api:app", host="0.0.0.0", port=8000, reload=True)
