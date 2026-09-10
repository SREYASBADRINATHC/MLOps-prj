"""
CartSense — Synthetic Electronics Dataset Generator

Generates a realistic dataset:
  - 1000 products  (600 laptops + 400 smartphones)
  - 500  users     with 8 preference segments
  - 10,000+ interactions with temporal structure suitable for ALS training
    and time-based train/test splits (concept drift demonstration)

All specification text is realistic and meaningful — not random gibberish.
The interaction matrix has enough structure for ALS to learn latent preferences.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_data")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@postgres_db:5432/cartsense",
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Laptop catalogue templates ──────────────────────────────────────────────

LAPTOP_BRANDS = ["Dell", "Lenovo", "ASUS", "HP", "Apple", "Acer", "MSI", "Razer", "LG", "Samsung"]

LAPTOP_LINES = {
    "Dell": ["XPS 15", "XPS 13", "Inspiron 15", "Vostro 14", "Alienware m16", "Latitude 7440"],
    "Lenovo": ["ThinkPad X1 Carbon", "ThinkPad T14s", "IdeaPad 5 Pro", "Legion Pro 5", "Yoga 9i", "LOQ 15"],
    "ASUS": ["ZenBook Pro 14", "ROG Zephyrus G16", "ProArt Studiobook 16", "VivoBook 15", "TUF Gaming A15", "ExpertBook B9"],
    "HP": ["Spectre x360 14", "Envy 16", "OMEN 16", "Pavilion 15", "EliteBook 840", "ZBook Firefly"],
    "Apple": ["MacBook Pro 14 M4", "MacBook Pro 16 M4 Max", "MacBook Air 13 M3", "MacBook Air 15 M3", "MacBook Pro 14 M3"],
    "Acer": ["Swift Go 14", "Predator Helios 16", "Aspire 5", "ConceptD 9", "Nitro 5", "Chromebook Spin 714"],
    "MSI": ["Titan GT77", "Raider GE78 HX", "Creator Z16P", "Prestige 16", "Katana 15", "Modern 14"],
    "Razer": ["Blade 14", "Blade 15", "Blade 16", "Blade 18"],
    "LG": ["Gram 14", "Gram 16", "Gram Pro 16", "Gram SuperSlim 15"],
    "Samsung": ["Galaxy Book4 Pro", "Galaxy Book4 Ultra", "Galaxy Book4 360", "Galaxy Book3 Pro"],
}

LAPTOP_PROCESSORS = [
    ("Intel Core Ultra 9 185H", "intel", "performance"),
    ("Intel Core Ultra 7 165H", "intel", "performance"),
    ("Intel Core Ultra 5 125H", "intel", "mainstream"),
    ("Intel Core Ultra 7 155U", "intel", "ultrabook"),
    ("Intel Core i9-13900HX", "intel", "performance"),
    ("Intel Core i7-13700H", "intel", "performance"),
    ("Intel Core i5-13500H", "intel", "mainstream"),
    ("AMD Ryzen 9 7945HX", "amd", "performance"),
    ("AMD Ryzen 7 7745HX", "amd", "performance"),
    ("AMD Ryzen 5 7535HS", "amd", "mainstream"),
    ("AMD Ryzen 9 8945HS AI", "amd_npu", "performance"),
    ("AMD Ryzen 7 8845HS AI", "amd_npu", "performance"),
    ("AMD Ryzen AI 9 HX 370", "amd_npu", "performance"),
    ("Apple M3", "apple", "performance"),
    ("Apple M3 Pro", "apple", "performance"),
    ("Apple M3 Max", "apple", "performance"),
    ("Apple M4", "apple", "performance"),
    ("Apple M4 Pro", "apple", "performance"),
    ("Apple M4 Max", "apple", "performance"),
    ("Qualcomm Snapdragon X Elite", "snapdragon", "performance"),
    ("Qualcomm Snapdragon X Plus", "snapdragon", "mainstream"),
]

LAPTOP_GPUS = [
    ("NVIDIA GeForce RTX 4090", "flagship"),
    ("NVIDIA GeForce RTX 4080", "flagship"),
    ("NVIDIA GeForce RTX 4070", "high_end"),
    ("NVIDIA GeForce RTX 4060", "mid"),
    ("NVIDIA GeForce RTX 4050", "mid"),
    ("NVIDIA GeForce RTX 3060", "mid"),
    ("AMD Radeon RX 7600M XT", "mid"),
    ("AMD Radeon RX 7700S", "mid"),
    ("Intel Arc A770M", "mid"),
    ("Apple M3 Max GPU", "integrated"),
    ("Apple M4 Pro GPU", "integrated"),
    ("Intel Iris Xe Graphics", "integrated"),
    ("AMD Radeon 780M", "integrated"),
    (None, "none"),
]

LAPTOP_DISPLAYS = ["OLED", "IPS", "Liquid Retina XDR", "Mini-LED", "IPS LCD", "AMOLED"]
LAPTOP_REFRESH = [60, 90, 120, 144, 165, 240]
LAPTOP_RAM = [8, 16, 24, 32, 48, 64, 96]
LAPTOP_STORAGE = [256, 512, 1024, 2048, 4096]
LAPTOP_BATTERY = [50, 60, 70, 75, 80, 90, 100, 120]
LAPTOP_OS = ["Windows 11 Home", "Windows 11 Pro", "macOS Sonoma", "Linux", "Chrome OS"]
LAPTOP_PRICE_BASE = {
    "budget": (499, 799),
    "mainstream": (799, 1299),
    "performance": (1299, 2199),
    "flagship": (2199, 4999),
}

# ── Smartphone catalogue templates ─────────────────────────────────────────────

PHONE_BRANDS = ["Samsung", "Apple", "Google", "OnePlus", "Xiaomi", "OPPO", "Vivo", "Sony", "Motorola", "Nothing"]

PHONE_LINES = {
    "Samsung": ["Galaxy S24 Ultra", "Galaxy S24+", "Galaxy S24", "Galaxy S23 FE", "Galaxy A55", "Galaxy A35", "Galaxy Z Fold 6", "Galaxy Z Flip 6"],
    "Apple": ["iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16", "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone SE 3rd Gen"],
    "Google": ["Pixel 9 Pro XL", "Pixel 9 Pro", "Pixel 9", "Pixel 8a", "Pixel 8 Pro", "Pixel 8", "Pixel Fold"],
    "OnePlus": ["12", "12R", "Open", "11", "Nord 4", "Nord CE 4", "10 Pro"],
    "Xiaomi": ["14 Ultra", "14 Pro", "14", "13T Pro", "POCO F6 Pro", "POCO X6 Pro", "Redmi Note 13 Pro+"],
    "OPPO": ["Find X7 Ultra", "Find X7", "Reno 12 Pro", "Reno 11 Pro", "A79", "F25 Pro"],
    "Vivo": ["X100 Ultra", "X100 Pro", "V30 Pro", "V30", "Y200 Pro", "iQOO 12"],
    "Sony": ["Xperia 1 VI", "Xperia 5 VI", "Xperia 10 VI", "Xperia 1 V"],
    "Motorola": ["Edge 50 Ultra", "Edge 50 Pro", "Moto G85", "Moto G Power 5G"],
    "Nothing": ["Phone (2a) Plus", "Phone (2a)", "Phone (2)", "CMF Phone 1"],
}

PHONE_CHIPSETS = [
    ("Snapdragon 8 Gen 3", "qualcomm", "flagship"),
    ("Snapdragon 8 Gen 2", "qualcomm", "flagship"),
    ("Snapdragon 8s Gen 3", "qualcomm", "upper_mid"),
    ("Snapdragon 7s Gen 2", "qualcomm", "mid"),
    ("Snapdragon 6 Gen 1", "qualcomm", "mid"),
    ("MediaTek Dimensity 9300+", "mediatek", "flagship"),
    ("MediaTek Dimensity 9300", "mediatek", "flagship"),
    ("MediaTek Dimensity 9200+", "mediatek", "flagship"),
    ("MediaTek Dimensity 8300 Ultra", "mediatek", "upper_mid"),
    ("MediaTek Dimensity 7300", "mediatek", "mid"),
    ("Apple A18 Pro", "apple", "flagship"),
    ("Apple A18", "apple", "flagship"),
    ("Apple A17 Pro", "apple", "flagship"),
    ("Apple A16 Bionic", "apple", "upper_mid"),
    ("Google Tensor G4", "google", "flagship"),
    ("Google Tensor G3", "google", "flagship"),
    ("Exynos 2400", "samsung", "flagship"),
    ("Exynos 2200", "samsung", "upper_mid"),
]

PHONE_DISPLAYS = ["AMOLED", "OLED", "Super AMOLED", "LTPO AMOLED", "Liquid Retina", "IPS LCD"]
PHONE_REFRESH = [60, 90, 120, 144]
PHONE_RAM = [4, 6, 8, 12, 16, 24]
PHONE_STORAGE = [64, 128, 256, 512, 1024]
PHONE_BATTERY = [3500, 4000, 4500, 5000, 5100, 5500]
PHONE_CAMERA = [12, 50, 64, 108, 200]
PHONE_OS = ["Android 14", "Android 13", "iOS 18", "iOS 17"]


# ── User segments ─────────────────────────────────────────────────────────────

USER_SEGMENTS = [
    "gaming",
    "productivity",
    "premium",
    "budget",
    "photography",
    "battery_focused",
    "high_refresh",
    "ai_npu",
]

# Segment → preferred features for interaction weighting
SEGMENT_PREFERENCES: dict[str, dict[str, Any]] = {
    "gaming": {
        "display_pref": ["OLED", "IPS", "AMOLED"],
        "refresh_min": 120,
        "ram_min": 16,
        "price_max": 3000,
        "price_min": 800,
    },
    "productivity": {
        "display_pref": ["IPS", "OLED", "Liquid Retina XDR"],
        "refresh_min": 60,
        "ram_min": 16,
        "price_max": 2500,
        "price_min": 600,
    },
    "premium": {
        "display_pref": ["OLED", "Liquid Retina XDR", "AMOLED", "Mini-LED"],
        "refresh_min": 90,
        "ram_min": 24,
        "price_max": 5000,
        "price_min": 1500,
    },
    "budget": {
        "display_pref": ["IPS", "AMOLED", "IPS LCD"],
        "refresh_min": 60,
        "ram_min": 8,
        "price_max": 900,
        "price_min": 200,
    },
    "photography": {
        "display_pref": ["OLED", "AMOLED", "LTPO AMOLED", "Super AMOLED"],
        "refresh_min": 60,
        "ram_min": 8,
        "price_max": 1500,
        "price_min": 600,
    },
    "battery_focused": {
        "display_pref": ["IPS", "AMOLED", "IPS LCD"],
        "refresh_min": 60,
        "ram_min": 8,
        "price_max": 1200,
        "price_min": 300,
    },
    "high_refresh": {
        "display_pref": ["OLED", "AMOLED", "LTPO AMOLED", "IPS"],
        "refresh_min": 120,
        "ram_min": 12,
        "price_max": 2000,
        "price_min": 500,
    },
    "ai_npu": {
        "display_pref": ["OLED", "AMOLED", "Liquid Retina XDR"],
        "refresh_min": 90,
        "ram_min": 16,
        "price_max": 3000,
        "price_min": 800,
    },
}


# ── Product generation ────────────────────────────────────────────────────────

def _make_laptop_spec_text(row: dict) -> str:
    parts = [
        row["name"],
        row.get("processor", ""),
        f"{int(row.get('ram_gb', 0))}GB RAM",
        f"{int(row.get('storage_gb', 0))}GB SSD",
        row.get("display_type", ""),
        f"{row.get('refresh_rate_hz', 60)}Hz",
        row.get("gpu") or "Integrated Graphics",
        f"{int(row.get('battery_mah', 60))}Wh battery",
        row.get("os", ""),
        f"${int(row.get('price_usd', 999))}",
    ]
    return " ".join(p for p in parts if p)


def _make_phone_spec_text(row: dict) -> str:
    parts = [
        row["name"],
        row.get("chipset", ""),
        f"{int(row.get('ram_gb', 0))}GB RAM",
        f"{int(row.get('storage_gb', 0))}GB",
        row.get("display_type", ""),
        f"{row.get('refresh_rate_hz', 60)}Hz",
        f"{int(row.get('battery_mah', 4000))}mAh",
        f"{int(row.get('camera_mp', 50))}MP camera",
        row.get("os", ""),
        f"${int(row.get('price_usd', 699))}",
    ]
    return " ".join(p for p in parts if p)


def generate_laptops(n: int = 600) -> list[dict]:
    """Generate n realistic laptop product records."""
    products = []
    idx = 0
    while len(products) < n:
        brand = random.choice(LAPTOP_BRANDS)
        line_options = LAPTOP_LINES.get(brand, ["Laptop"])
        line = random.choice(line_options)

        tier = random.choices(
            ["budget", "mainstream", "performance", "flagship"],
            weights=[0.15, 0.40, 0.30, 0.15],
        )[0]

        proc_options = LAPTOP_PROCESSORS
        if brand == "Apple":
            proc_options = [p for p in LAPTOP_PROCESSORS if p[1] == "apple"]
        elif tier == "flagship":
            proc_options = [p for p in LAPTOP_PROCESSORS if p[2] == "performance"]
        elif tier == "budget":
            proc_options = [p for p in LAPTOP_PROCESSORS if p[2] in ("mainstream", "ultrabook")]

        processor_tuple = random.choice(proc_options)
        processor = processor_tuple[0]
        proc_family = processor_tuple[1]
        is_npu = proc_family in ("amd_npu", "snapdragon") or "AI" in processor or "NPU" in processor

        ram = random.choice(LAPTOP_RAM if tier != "budget" else [8, 16])
        storage = random.choice(LAPTOP_STORAGE if tier != "budget" else [256, 512])
        display = random.choice(LAPTOP_DISPLAYS)
        refresh = random.choices(
            LAPTOP_REFRESH,
            weights=[0.20, 0.05, 0.30, 0.25, 0.10, 0.10],
        )[0]
        if tier == "budget":
            refresh = random.choice([60, 90])
            display = random.choice(["IPS", "IPS LCD"])

        gpu_options = LAPTOP_GPUS
        if brand == "Apple":
            gpu_options = [g for g in LAPTOP_GPUS if "Apple" in (g[0] or "")]
        elif tier == "budget":
            gpu_options = [g for g in LAPTOP_GPUS if g[1] in ("integrated", "none")]
        elif tier == "flagship":
            gpu_options = [g for g in LAPTOP_GPUS if g[1] in ("flagship", "high_end")]
        gpu_tuple = random.choice(gpu_options)
        gpu = gpu_tuple[0]

        battery = random.choice(LAPTOP_BATTERY)
        os_name = "macOS Sonoma" if brand == "Apple" else random.choice(LAPTOP_OS[:4])

        pmin, pmax = LAPTOP_PRICE_BASE[tier]
        price = round(random.uniform(pmin, pmax), -1)

        gen = random.randint(1, 4)
        full_name = f"{brand} {line} Gen {gen}"

        product_id = f"L{idx + 1:04d}"
        row: dict[str, Any] = {
            "product_id": product_id,
            "category": "laptop",
            "brand": brand,
            "name": full_name,
            "description": (
                f"The {full_name} is a {tier}-tier laptop powered by the {processor}. "
                f"Featuring {ram}GB RAM and {storage}GB SSD with a {display} {refresh}Hz display."
            ),
            "processor": processor,
            "ram_gb": float(ram),
            "storage_gb": float(storage),
            "display_type": display,
            "refresh_rate_hz": refresh,
            "gpu": gpu,
            "battery_mah": battery,
            "camera_mp": None,
            "chipset": None,
            "os": os_name,
            "price_usd": price,
            "interaction_count": 0,
        }
        row["spec_text"] = _make_laptop_spec_text(row)
        products.append(row)
        idx += 1
    return products


def generate_smartphones(n: int = 400, offset: int = 600) -> list[dict]:
    """Generate n realistic smartphone product records."""
    products = []
    idx = 0
    while len(products) < n:
        brand = random.choice(PHONE_BRANDS)
        line_options = PHONE_LINES.get(brand, ["Phone"])
        line = random.choice(line_options)

        tier = random.choices(
            ["budget", "mid", "upper_mid", "flagship"],
            weights=[0.15, 0.30, 0.30, 0.25],
        )[0]

        chip_options = PHONE_CHIPSETS
        if brand == "Apple":
            chip_options = [c for c in PHONE_CHIPSETS if c[1] == "apple"]
        elif brand == "Google":
            chip_options = [c for c in PHONE_CHIPSETS if c[1] == "google"]
        elif tier == "flagship":
            chip_options = [c for c in PHONE_CHIPSETS if c[2] == "flagship"]
        elif tier == "budget":
            chip_options = [c for c in PHONE_CHIPSETS if c[2] in ("mid",)]

        chipset_tuple = random.choice(chip_options)
        chipset = chipset_tuple[0]

        ram = random.choice(PHONE_RAM if tier != "budget" else [4, 6, 8])
        storage = random.choice(PHONE_STORAGE if tier != "budget" else [64, 128])
        display = random.choice(PHONE_DISPLAYS)
        refresh = random.choices(
            PHONE_REFRESH,
            weights=[0.15, 0.20, 0.55, 0.10],
        )[0]
        if tier == "budget":
            refresh = 60
            display = random.choice(["IPS LCD", "AMOLED"])

        battery = random.choice(PHONE_BATTERY)
        camera = random.choice(PHONE_CAMERA if tier != "budget" else [12, 50])
        os_name = "iOS 18" if brand == "Apple" else random.choice(PHONE_OS[:2])

        price_ranges = {
            "budget": (149, 399),
            "mid": (399, 699),
            "upper_mid": (699, 999),
            "flagship": (999, 1599),
        }
        pmin, pmax = price_ranges[tier]
        price = round(random.uniform(pmin, pmax), -1)

        full_name = f"{brand} {line}"

        product_id = f"S{idx + 1:04d}"
        row: dict[str, Any] = {
            "product_id": product_id,
            "category": "smartphone",
            "brand": brand,
            "name": full_name,
            "description": (
                f"The {full_name} runs on {chipset} with {ram}GB RAM. "
                f"Features {display} {refresh}Hz display and {camera}MP camera."
            ),
            "processor": None,
            "ram_gb": float(ram),
            "storage_gb": float(storage),
            "display_type": display,
            "refresh_rate_hz": refresh,
            "gpu": None,
            "battery_mah": battery,
            "camera_mp": camera,
            "chipset": chipset,
            "os": os_name,
            "price_usd": price,
            "interaction_count": 0,
        }
        row["spec_text"] = _make_phone_spec_text(row)
        products.append(row)
        idx += 1
    return products


# ── User generation ───────────────────────────────────────────────────────────

def generate_users(n: int = 500) -> list[dict]:
    """Generate n users with preference segments."""
    users = []
    for i in range(1, n + 1):
        segment = random.choice(USER_SEGMENTS)
        users.append({"user_id": f"U{i:04d}", "segment": segment})
    return users


# ── Interaction generation ────────────────────────────────────────────────────

INTERACTION_TYPES = ["view", "click", "wishlist", "cart", "purchase"]
INTERACTION_WEIGHTS = {"view": 1.0, "click": 2.0, "wishlist": 3.0, "cart": 4.0, "purchase": 5.0}

def _product_score_for_segment(product: dict, segment: str) -> float:
    """Compute affinity score between a product and user segment [0-1]."""
    prefs = SEGMENT_PREFERENCES.get(segment, {})
    score = 0.0
    factors = 0

    display = product.get("display_type", "")
    if display in prefs.get("display_pref", []):
        score += 1.0
    factors += 1

    refresh = product.get("refresh_rate_hz", 60)
    if refresh >= prefs.get("refresh_min", 60):
        score += 1.0
    factors += 1

    ram = product.get("ram_gb", 8)
    if ram >= prefs.get("ram_min", 8):
        score += 1.0
    factors += 1

    price = product.get("price_usd", 999)
    pmin = prefs.get("price_min", 0)
    pmax = prefs.get("price_max", 9999)
    if pmin <= price <= pmax:
        score += 1.0
    elif price < pmin:
        score += 0.3
    else:
        score += 0.1
    factors += 1

    return score / factors


def generate_interactions(
    users: list[dict],
    products: list[dict],
    n_target: int = 12000,
    now: datetime | None = None,
) -> list[dict]:
    """
    Generate temporally-structured interactions.

    Design:
    - First 60% of the time window (days 0-180): baseline preferences
      (IPS, 60Hz, modest RAM — older market)
    - Last 40% of the time window (days 180-300): shifted preferences
      (OLED, 120Hz+, 32GB+ RAM — newer market/drifted)
    This allows temporal train/test split for drift evaluation.
    """
    if now is None:
        now = datetime.utcnow()

    # Build product index
    product_df = pd.DataFrame(products)
    laptop_ids = product_df[product_df["category"] == "laptop"]["product_id"].tolist()
    phone_ids = product_df[product_df["category"] == "smartphone"]["product_id"].tolist()
    prod_by_id: dict[str, dict] = {p["product_id"]: p for p in products}

    # Each user has a category preference (laptop vs phone vs both)
    user_cat_pref: dict[str, str] = {}
    for u in users:
        user_cat_pref[u["user_id"]] = random.choices(
            ["laptop", "smartphone", "both"], weights=[0.35, 0.35, 0.30]
        )[0]

    interactions = []
    interaction_counter = 0

    target_per_user = max(20, n_target // len(users))

    for user in users:
        uid = user["user_id"]
        segment = user["segment"]
        cat_pref = user_cat_pref[uid]

        # Candidate pool for this user
        if cat_pref == "laptop":
            candidates = laptop_ids
        elif cat_pref == "smartphone":
            candidates = phone_ids
        else:
            candidates = laptop_ids + phone_ids

        # Score all candidates for this segment
        scores = np.array([_product_score_for_segment(prod_by_id[pid], segment) for pid in candidates])
        # Add noise
        scores = scores + np.random.exponential(0.15, len(scores))
        scores = np.clip(scores, 0.01, None)
        probs = scores / scores.sum()

        # Number of unique products this user interacts with
        n_unique = min(random.randint(15, 40), len(candidates))
        chosen_indices = np.random.choice(len(candidates), size=n_unique, replace=False, p=probs)
        chosen_products = [candidates[i] for i in chosen_indices]

        for pid in chosen_products:
            # Each product may generate 1-4 interaction events
            n_events = random.choices([1, 2, 3, 4], weights=[0.40, 0.30, 0.20, 0.10])[0]
            prod = prod_by_id[pid]
            affinity = _product_score_for_segment(prod, segment)

            # Temporal placement: newer/drifted products cluster in later window
            is_drifted_product = (
                prod.get("display_type") in ("OLED", "AMOLED", "LTPO AMOLED")
                and prod.get("refresh_rate_hz", 60) >= 120
                and prod.get("ram_gb", 0) >= 24
            )
            if is_drifted_product:
                # More likely in recent window (days 0-120 from now going backward)
                day_offset = random.randint(0, 120)
            else:
                # Can be anywhere
                day_offset = random.randint(0, 300)

            ts_base = now - timedelta(days=day_offset)

            for event_idx in range(n_events):
                itype = random.choices(
                    INTERACTION_TYPES,
                    weights=[0.40, 0.25, 0.15, 0.12, 0.08],
                )[0]
                # Higher affinity → better event types
                if affinity > 0.75:
                    itype = random.choices(
                        INTERACTION_TYPES,
                        weights=[0.10, 0.20, 0.25, 0.25, 0.20],
                    )[0]
                weight = INTERACTION_WEIGHTS[itype]
                rating = None
                if itype in ("purchase", "rating"):
                    rating = round(2.5 + 2.5 * affinity + random.gauss(0, 0.3), 1)
                    rating = float(np.clip(rating, 1.0, 5.0))

                ts = ts_base + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
                interactions.append(
                    {
                        "user_id": uid,
                        "product_id": pid,
                        "interaction_type": itype,
                        "rating": rating,
                        "weight": weight,
                        "timestamp": ts,
                    }
                )
                interaction_counter += 1

    # Shuffle temporal order
    random.shuffle(interactions)
    logger.info("Generated %d interactions for %d users", len(interactions), len(users))
    return interactions


# ── Database persistence ──────────────────────────────────────────────────────

def get_engine():
    is_sqlite = DATABASE_URL.startswith("sqlite")
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 3
        kwargs["max_overflow"] = 5
    return create_engine(DATABASE_URL, **kwargs)


def init_schema(eng) -> None:
    """Create all tables using ORM metadata (idempotent)."""
    # Import models to register them against Base
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from backend.models import (  # noqa: F401
        Product, User, Interaction, PredictionLog,
        TrainingRun, DriftMetric, ModelRegistryMetadata
    )
    from backend.database import Base
    Base.metadata.create_all(bind=eng)
    logger.info("Database schema initialized.")


def seed_database(products: list[dict], users: list[dict], interactions: list[dict], eng) -> None:
    """Insert all data into the database."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    with eng.begin() as conn:
        # --- Products ---
        conn.execute(text("DELETE FROM products"))
        for row in products:
            conn.execute(
                text("""
                    INSERT INTO products
                      (product_id, category, brand, name, description, spec_text,
                       processor, ram_gb, storage_gb, display_type, refresh_rate_hz,
                       gpu, battery_mah, camera_mp, chipset, os, price_usd,
                       interaction_count, created_at, updated_at)
                    VALUES
                      (:product_id, :category, :brand, :name, :description, :spec_text,
                       :processor, :ram_gb, :storage_gb, :display_type, :refresh_rate_hz,
                       :gpu, :battery_mah, :camera_mp, :chipset, :os, :price_usd,
                       0, NOW(), NOW())
                """),
                row,
            )
        logger.info("Inserted %d products.", len(products))

        # --- Users ---
        conn.execute(text("DELETE FROM users"))
        for row in users:
            conn.execute(
                text("INSERT INTO users (user_id, segment, created_at) VALUES (:user_id, :segment, NOW())"),
                row,
            )
        logger.info("Inserted %d users.", len(users))

        # --- Interactions ---
        conn.execute(text("DELETE FROM interactions"))
        for row in interactions:
            conn.execute(
                text("""
                    INSERT INTO interactions
                      (user_id, product_id, interaction_type, rating, weight, timestamp)
                    VALUES
                      (:user_id, :product_id, :interaction_type, :rating, :weight, :timestamp)
                """),
                row,
            )
        logger.info("Inserted %d interactions.", len(interactions))

        # --- Update interaction_count per product ---
        conn.execute(
            text("""
                UPDATE products p
                SET interaction_count = sub.cnt,
                    updated_at = NOW()
                FROM (
                    SELECT product_id, COUNT(*) AS cnt
                    FROM interactions
                    GROUP BY product_id
                ) sub
                WHERE p.product_id = sub.product_id
            """)
        )
        logger.info("Updated product interaction counts.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CartSense data generator")
    parser.add_argument("--n-laptops", type=int, default=600)
    parser.add_argument("--n-phones", type=int, default=400)
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--n-interactions", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Generating %d laptops...", args.n_laptops)
    laptops = generate_laptops(args.n_laptops)

    logger.info("Generating %d smartphones...", args.n_phones)
    phones = generate_smartphones(args.n_phones)

    all_products = laptops + phones
    logger.info("Total products: %d", len(all_products))

    logger.info("Generating %d users...", args.n_users)
    users = generate_users(args.n_users)

    logger.info("Generating interactions (target=%d)...", args.n_interactions)
    interactions = generate_interactions(users, all_products, n_target=args.n_interactions)

    logger.info("Connecting to database...")
    eng = get_engine()

    logger.info("Initializing schema...")
    init_schema(eng)

    logger.info("Seeding database...")
    seed_database(all_products, users, interactions, eng)

    print(f"✅ Seeded: {len(all_products)} products, {len(users)} users, {len(interactions)} interactions")


if __name__ == "__main__":
    main()
