import os
import pandas as pd
from sqlalchemy import create_engine

# Database Connection URI from environment variable or defaulting to docker-compose config
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres_secret_password@localhost:5432/feature_store"
)

def run_ingestion():
    print("🚀 Starting Data Ingestion Pipeline...")
    
    # 1. Create Mock Products DataFrame (5 Laptops, 5 Mobile Phones)
    products_data = {
        "product_id": [
            "laptop_x1", "laptop_macbook", "laptop_xps13", "laptop_rog", "laptop_latitude",
            "phone_iphone15", "phone_s24", "phone_pixel8", "phone_oneplus12", "phone_nothing2"
        ],
        "name": [
            "ThinkPad X1 Carbon (Gen 12)", "MacBook Pro 14\" M3", "Dell XPS 13 (2026 Model)", 
            "ROG Zephyrus G14", "Dell Latitude 5440", "iPhone 15 Pro Max", 
            "Samsung Galaxy S24 Ultra", "Google Pixel 8 Pro", "OnePlus 12 5G", "Nothing Phone (2)"
        ],
        "specs": [
            "Intel Core Ultra 7, 32GB LPDDR5X RAM, 1TB NVMe PCIe Gen4 SSD, 14\" WUXGA IPS Display",
            "Apple M3 Pro Chip with 11‑core CPU, 14‑core GPU, 18GB Unified Memory, 512GB SSD",
            "Intel Core Ultra 7, 16GB LPDDR5x, 512GB SSD, 13.4\" FHD+ InfinityEdge Non-Touch",
            "AMD Ryzen 9 8945HS, NVIDIA GeForce RTX 4070, 16GB DDR5, 1TB SSD, 14\" ROG Nebula OLED",
            "Intel Core i5-1335U, 16GB DDR4 RAM, 512GB PCIe NVMe SSD, 14\" FHD Anti-Glare",
            "A17 Pro Chip, Triple Camera System (48MP Main), 256GB, Super Retina XDR OLED",
            "Snapdragon 8 Gen 3, Quad Telephoto Zoom, 12GB RAM, 256GB, Dynamic AMOLED 2X",
            "Google Tensor G3, Pro Triple Camera System, Magic Editor AI, 128GB, Super Actua Display",
            "Snapdragon 8 Gen 3, 4th Gen Hasselblad Camera, 16GB RAM, 512GB, 120Hz 2K AMOLED",
            "Snapdragon 8+ Gen 1, Glyph Interface 2.0, Dual 50MP Cameras, 12GB RAM, 256GB OLED"
        ],
        "category": [
            "Laptop", "Laptop", "Laptop", "Laptop", "Laptop",
            "Mobile Phone", "Mobile Phone", "Mobile Phone", "Mobile Phone", "Mobile Phone"
        ]
    }
    df_products = pd.DataFrame(products_data)
    print(f"📊 Created Products DataFrame ({len(df_products)} records):")
    print(df_products[["product_id", "name", "category"]])

    # 2. Create Mock Interaction DataFrame (user_id, product_id, rating)
    # We will simulate ratings for some items, leaving others with 0 interactions to demonstrate Cold Start.
    interactions_data = {
        "user_id": [
            101, 101, 102, 102, 103, 
            103, 104, 104, 105, 105
        ],
        "product_id": [
            "laptop_x1", "phone_iphone15", "laptop_macbook", "phone_s24", "laptop_rog",
            "phone_pixel8", "laptop_latitude", "phone_oneplus12", "laptop_x1", "laptop_macbook"
        ],
        "rating": [
            5.0, 4.5, 4.0, 5.0, 4.5,
            4.0, 3.5, 4.5, 5.0, 4.5
        ]
    }
    df_interactions = pd.DataFrame(interactions_data)
    print(f"\n📊 Created Interactions DataFrame ({len(df_interactions)} records):")
    print(df_interactions)

    # 3. Connection and Table Insertion
    try:
        print(f"\n🔌 Connecting to database at: {DATABASE_URL.split('@')[-1]}")
        engine = create_engine(DATABASE_URL)
        
        # Write DataFrames to PostgreSQL
        # If table exists, we replace it with updated mock data
        df_products.to_sql("products", con=engine, if_exists="replace", index=False)
        print("✅ Pushed 'products' table to Feature Store successfully.")

        df_interactions.to_sql("interactions", con=engine, if_exists="replace", index=False)
        print("✅ Pushed 'interactions' table to Feature Store successfully.")
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {str(e)}")
        print("💡 Ensure that PostgreSQL is running and accessible (run 'docker-compose up db' first).")

if __name__ == "__main__":
    run_ingestion()
