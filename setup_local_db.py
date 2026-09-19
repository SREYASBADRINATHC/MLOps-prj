"""
Local development setup script for CartSense.
Configures SQLite database with all required tables and sample data,
then starts the FastAPI backend.
"""
import os
import sys
import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("setup")

DB_PATH = "cartsense_local.db"

def setup_database():
    """Set up a complete local SQLite database with all required tables."""
    logger.info(f"Setting up local SQLite database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create all required tables
    c.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            brand TEXT,
            name TEXT NOT NULL,
            description TEXT,
            spec_text TEXT,
            processor TEXT,
            ram_gb INTEGER,
            storage_gb INTEGER,
            display_type TEXT,
            refresh_rate_hz INTEGER,
            gpu TEXT,
            battery_mah INTEGER,
            camera_mp INTEGER,
            chipset TEXT,
            os TEXT,
            price_usd REAL,
            image_url TEXT,
            product_url TEXT,
            release_date TEXT,
            is_upcoming INTEGER DEFAULT 0,
            interaction_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            segment TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            product_id TEXT,
            event_type TEXT,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            user_id TEXT,
            product_id TEXT,
            recommendation_mode TEXT,
            model_version TEXT,
            top_k INTEGER,
            latency_ms REAL,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            model_type TEXT,
            model_version TEXT,
            rmse REAL,
            precision_at_k REAL,
            recall_at_k REAL,
            ndcg_at_k REAL,
            status TEXT,
            promoted INTEGER DEFAULT 0,
            timestamp TEXT
        );

        CREATE TABLE IF NOT EXISTS drift_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            dataset_drift_detected INTEGER DEFAULT 0,
            share_drifted_columns REAL,
            n_drifted_columns INTEGER,
            n_total_columns INTEGER,
            drift_threshold REAL,
            retraining_recommended INTEGER DEFAULT 0,
            simulation_mode INTEGER DEFAULT 0,
            column_results TEXT,
            timestamp TEXT
        );
    """)

    # Check if products already exist
    c.execute("SELECT COUNT(*) FROM products")
    if c.fetchone()[0] == 0:
        logger.info("Inserting sample products...")
        products = [
            ("P001", "laptop", "Apple", "MacBook Pro 16\" M3 Max",
             "Apple Silicon powerhouse for professionals", 
             "M3 Max 16-core CPU, 40-core GPU, 128GB RAM, 1TB SSD, Liquid Retina XDR 16.2\"",
             "Apple M3 Max", 128, 1000, "OLED", 120, "Integrated 40-core GPU", None, None,
             "Apple M3 Max", "macOS Sequoia", 3499, None, None, None, 0, 45),
            ("P002", "laptop", "Dell", "XPS 15 (2024)",
             "Premium Windows laptop with OLED display",
             "Intel Core Ultra 9 185H, RTX 4070, 32GB DDR5, 1TB SSD, 15.6\" OLED 4K 120Hz",
             "Intel Core Ultra 9 185H", 32, 1000, "OLED", 120, "NVIDIA RTX 4070", None, None,
             "Intel Core Ultra 9", "Windows 11", 2199, None, None, None, 0, 38),
            ("P003", "laptop", "ASUS", "ROG Zephyrus G16 (2024)",
             "Gaming powerhouse with AMD Ryzen AI",
             "AMD Ryzen AI 9 HX 370, RTX 4090, 32GB DDR5, 2TB SSD, 16\" QHD+ OLED 240Hz",
             "AMD Ryzen AI 9 HX 370", 32, 2000, "OLED", 240, "NVIDIA RTX 4090", None, None,
             "AMD Ryzen AI 9", "Windows 11", 2799, None, None, None, 0, 52),
            ("P004", "laptop", "Lenovo", "ThinkPad X1 Carbon Gen 13",
             "Ultra-light business laptop with AI features",
             "Intel Core Ultra 7 155H, 32GB LPDDR5, 1TB SSD, 14\" 2.8K OLED 120Hz, 1.09kg",
             "Intel Core Ultra 7 155H", 32, 1000, "OLED", 120, "Intel Iris Xe", None, None,
             "Intel Core Ultra 7", "Windows 11 Pro", 1899, None, None, None, 0, 29),
            ("P005", "smartphone", "Apple", "iPhone 16 Pro Max",
             "Apple's most advanced iPhone with A18 Pro chip",
             "A18 Pro, 8GB RAM, 256GB, 6.9\" Super Retina XDR ProMotion 120Hz, Camera Control, Titanium",
             "Apple A18 Pro", 8, 256, "Super Retina XDR OLED", 120, None, 4685, 48,
             "Apple A18 Pro", "iOS 18", 1199, None, None, None, 0, 67),
            ("P006", "smartphone", "Samsung", "Galaxy S25 Ultra",
             "Samsung flagship with Snapdragon 8 Elite and S-Pen",
             "Snapdragon 8 Elite, 12GB RAM, 512GB, 6.9\" QHD+ Dynamic AMOLED 120Hz, 200MP camera, S-Pen",
             "Snapdragon 8 Elite", 12, 512, "Dynamic AMOLED 2X", 120, None, 5000, 200,
             "Qualcomm Snapdragon 8 Elite", "Android 15", 1299, None, None, None, 0, 71),
            ("P007", "smartphone", "Google", "Pixel 9 Pro XL",
             "Google Pixel flagship with Tensor G4 and Gemini AI",
             "Tensor G4, 16GB RAM, 256GB, 6.8\" LTPO OLED 120Hz, 50MP triple camera, Gemini AI",
             "Google Tensor G4", 16, 256, "LTPO OLED", 120, None, 5060, 50,
             "Google Tensor G4", "Android 15", 1099, None, None, None, 0, 41),
            ("P008", "smartphone", "OnePlus", "OnePlus 13",
             "Premium Android with fast charging and Hasselblad cameras",
             "Snapdragon 8 Elite, 12GB RAM, 256GB, 6.82\" 2K AMOLED 120Hz, Hasselblad triple 50MP, 100W charging",
             "Snapdragon 8 Elite", 12, 256, "2K AMOLED", 120, None, 6000, 50,
             "Qualcomm Snapdragon 8 Elite", "Android 15", 899, None, None, None, 0, 35),
            ("P009", "laptop", "Microsoft", "Surface Laptop 7 (Snapdragon)",
             "Ultra-thin Copilot+ PC with Snapdragon X Elite",
             "Snapdragon X Elite, 32GB LPDDR5X, 1TB SSD, 15\" PixelSense 2496x1664 120Hz, 22h battery",
             "Snapdragon X Elite", 32, 1000, "PixelSense IPS", 120, "Qualcomm Adreno", None, None,
             "Qualcomm Snapdragon X Elite", "Windows 11", 1699, None, None, None, 0, 23),
            ("P010", "smartphone", "Xiaomi", "Xiaomi 15 Ultra",
             "Camera champion with Leica quad-camera system",
             "Snapdragon 8 Elite, 16GB RAM, 512GB, 6.73\" WQHD+ AMOLED 120Hz, Leica quad 200MP, 90W charging",
             "Snapdragon 8 Elite", 16, 512, "WQHD+ AMOLED", 120, None, 5410, 200,
             "Qualcomm Snapdragon 8 Elite", "Android 15", 1099, None, None, None, 0, 28),
        ]
        c.executemany("""
            INSERT OR IGNORE INTO products 
            (product_id, category, brand, name, description, spec_text, processor, 
             ram_gb, storage_gb, display_type, refresh_rate_hz, gpu, battery_mah, camera_mp,
             chipset, os, price_usd, image_url, product_url, release_date, is_upcoming, interaction_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, products)

    # Check if users exist
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        logger.info("Inserting sample users...")
        users = [
            ("U0001", "tech_enthusiast", datetime.utcnow().isoformat()),
            ("U0002", "gamer", datetime.utcnow().isoformat()),
            ("U0003", "creative", datetime.utcnow().isoformat()),
            ("U0004", "business", datetime.utcnow().isoformat()),
            ("U0005", "budget", datetime.utcnow().isoformat()),
        ]
        c.executemany("INSERT OR IGNORE INTO users (user_id, segment, created_at) VALUES (?,?,?)", users)

    # Check if interactions exist
    c.execute("SELECT COUNT(*) FROM interactions")
    if c.fetchone()[0] == 0:
        logger.info("Inserting sample interactions...")
        interactions = [
            ("U0001", "P001", "view", datetime.utcnow().isoformat()),
            ("U0001", "P005", "purchase", datetime.utcnow().isoformat()),
            ("U0001", "P002", "view", datetime.utcnow().isoformat()),
            ("U0002", "P003", "view", datetime.utcnow().isoformat()),
            ("U0002", "P003", "purchase", datetime.utcnow().isoformat()),
            ("U0002", "P006", "view", datetime.utcnow().isoformat()),
            ("U0003", "P001", "purchase", datetime.utcnow().isoformat()),
            ("U0003", "P007", "view", datetime.utcnow().isoformat()),
            ("U0004", "P004", "view", datetime.utcnow().isoformat()),
            ("U0004", "P004", "purchase", datetime.utcnow().isoformat()),
            ("U0005", "P008", "view", datetime.utcnow().isoformat()),
            ("U0005", "P010", "view", datetime.utcnow().isoformat()),
        ]
        c.executemany("INSERT INTO interactions (user_id, product_id, event_type, timestamp) VALUES (?,?,?,?)", interactions)

    # Insert a sample training run
    c.execute("SELECT COUNT(*) FROM training_runs")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO training_runs (run_id, model_type, model_version, rmse, precision_at_k, 
            recall_at_k, ndcg_at_k, status, promoted, timestamp)
            VALUES ('local-run-001', 'TF-IDF', 'v1.0.0-local', 0.42, 0.78, 0.65, 0.71, 
            'completed', 1, ?)
        """, (datetime.utcnow().isoformat(),))

    conn.commit()
    conn.close()
    logger.info("Database setup complete!")

if __name__ == "__main__":
    setup_database()
    print(f"✅ Local SQLite database ready at: {DB_PATH}")
    print(f"   Set DATABASE_URL=sqlite:///{DB_PATH} to use it.")
