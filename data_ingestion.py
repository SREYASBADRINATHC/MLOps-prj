import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@localhost:5432/cartsense",
)


def create_products_dataframe() -> pd.DataFrame:
    """Create a deterministic mock catalog for laptops and mobiles."""
    rows = [
        (101, "Apple MacBook Pro 16 M3 Max", "Laptop", "Apple", 3499, 48, 1024, "Liquid Retina XDR", "Apple M3 Max"),
        (102, "Dell XPS 15 OLED", "Laptop", "Dell", 2399, 32, 1024, "OLED", "Intel Core Ultra 9"),
        (103, "Lenovo ThinkPad X1 Carbon Gen 12", "Laptop", "Lenovo", 2199, 32, 1024, "IPS", "Intel Core Ultra 7"),
        (104, "ASUS ROG Zephyrus G16", "Laptop", "ASUS", 2599, 32, 2048, "OLED", "Intel Core Ultra 9"),
        (105, "HP Spectre x360 14", "Laptop", "HP", 1899, 16, 1024, "OLED", "Intel Core Ultra 7"),
        (201, "Samsung Galaxy S24 Ultra", "Mobile", "Samsung", 1299, 12, 512, "AMOLED", "Snapdragon 8 Gen 3"),
        (202, "Apple iPhone 15 Pro Max", "Mobile", "Apple", 1399, 8, 512, "OLED", "A17 Pro"),
        (203, "Google Pixel 9 Pro", "Mobile", "Google", 1099, 16, 256, "OLED", "Tensor G4"),
        (204, "OnePlus 12", "Mobile", "OnePlus", 899, 16, 512, "AMOLED", "Snapdragon 8 Gen 3"),
        (205, "Xiaomi 14 Ultra", "Mobile", "Xiaomi", 999, 16, 512, "AMOLED", "Snapdragon 8 Gen 3"),
    ]
    columns = [
        "product_id",
        "product_name",
        "category",
        "brand",
        "price_usd",
        "ram_gb",
        "storage_gb",
        "display_type",
        "chipset",
    ]
    products_df = pd.DataFrame(rows, columns=columns)
    products_df["specs_text"] = (
        products_df["display_type"]
        + ", "
        + products_df["ram_gb"].astype(str)
        + "GB RAM, "
        + products_df["storage_gb"].astype(str)
        + "GB storage, "
        + products_df["chipset"]
        + ", "
        + products_df["category"]
    )
    return products_df


def create_interactions_dataframe() -> pd.DataFrame:
    """Create deterministic user-product interactions with one true cold-start SKU."""
    now = datetime.utcnow()
    rows = [
        (1, 101, 5.0, "view", now - timedelta(days=10)),
        (1, 102, 4.5, "purchase", now - timedelta(days=8)),
        (1, 202, 4.0, "view", now - timedelta(days=6)),
        (2, 104, 5.0, "purchase", now - timedelta(days=9)),
        (2, 204, 4.0, "view", now - timedelta(days=5)),
        (2, 201, 4.5, "view", now - timedelta(days=4)),
        (3, 103, 4.0, "view", now - timedelta(days=12)),
        (3, 105, 4.2, "purchase", now - timedelta(days=7)),
        (3, 203, 4.8, "purchase", now - timedelta(days=2)),
        (4, 101, 4.6, "view", now - timedelta(days=11)),
        (4, 201, 4.4, "purchase", now - timedelta(days=7)),
        (5, 102, 4.9, "purchase", now - timedelta(days=10)),
        (5, 104, 4.8, "view", now - timedelta(days=3)),
        (5, 202, 4.3, "view", now - timedelta(days=1)),
    ]
    columns = ["user_id", "product_id", "rating", "event_type", "event_ts"]
    return pd.DataFrame(rows, columns=columns)


def write_dataframes_to_postgres(products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
    """Write DataFrames into PostgreSQL with connection drop resilience."""
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
    )

    try:
        with engine.begin() as connection:
            products_df.to_sql("products", con=connection, if_exists="replace", index=False)
            interactions_df.to_sql("interactions", con=connection, if_exists="replace", index=False)
    except OperationalError as exc:
        # Dispose stale pooled connections and retry exactly once.
        engine.dispose()
        try:
            with engine.begin() as connection:
                products_df.to_sql("products", con=connection, if_exists="replace", index=False)
                interactions_df.to_sql("interactions", con=connection, if_exists="replace", index=False)
        except SQLAlchemyError as retry_exc:
            raise RuntimeError(f"Retry failed while writing to PostgreSQL: {retry_exc}") from retry_exc
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Database write failed: {exc}") from exc
    finally:
        engine.dispose()


def main() -> None:
    products_df = create_products_dataframe()
    interactions_df = create_interactions_dataframe()
    write_dataframes_to_postgres(products_df, interactions_df)
    print(f"Loaded {len(products_df)} products and {len(interactions_df)} interactions into PostgreSQL.")


if __name__ == "__main__":
    main()
