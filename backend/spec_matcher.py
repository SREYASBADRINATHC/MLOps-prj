"""
CartSense — Specification Matching Engine

Parses structured hardware specifications from free-text queries and product records,
then computes a normalized similarity score between requirement and candidate.

Score is computed as a weighted average across matched attributes.
Each attribute returns:
  1.0  = exact / fully satisfied match
  0.5  = partial match (e.g. refresh rate within 20% of target)
  0.0  = no match / not present

The final spec_match_score is in [0, 1].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedSpecs:
    """Parsed hardware requirement/attributes."""
    ram_gb: Optional[float] = None
    storage_gb: Optional[float] = None
    display_type: Optional[str] = None          # OLED | AMOLED | IPS | LCD | Retina
    refresh_rate_hz: Optional[int] = None
    price_max_usd: Optional[float] = None
    price_min_usd: Optional[float] = None
    battery_min_mah: Optional[int] = None
    camera_min_mp: Optional[int] = None
    processor_keywords: list[str] = field(default_factory=list)

    @property
    def has_requirements(self) -> bool:
        return any([
            self.ram_gb,
            self.storage_gb,
            self.display_type,
            self.refresh_rate_hz,
            self.price_max_usd,
            self.battery_min_mah,
            self.camera_min_mp,
            self.processor_keywords,
        ])


# ── Parsing utilities ─────────────────────────────────────────────────────────

_RAM_PATTERN = re.compile(r"\b(\d+)\s*gb?\s*ram\b", re.IGNORECASE)
_STORAGE_TB_PATTERN = re.compile(r"\b(\d+)\s*tb\s*(?:ssd|nvme|storage|hdd)?\b", re.IGNORECASE)
_STORAGE_GB_PATTERN = re.compile(r"\b(\d+)\s*gb\s*(?:ssd|nvme|storage|hdd)\b", re.IGNORECASE)
_REFRESH_PATTERN = re.compile(r"\b(\d+)\s*hz\b", re.IGNORECASE)
_BATTERY_PATTERN = re.compile(r"\b(\d{3,5})\s*mah\b", re.IGNORECASE)
_CAMERA_PATTERN = re.compile(r"\b(\d+)\s*mp\b", re.IGNORECASE)
_PRICE_UNDER_PATTERN = re.compile(r"under\s*\$?(\d+)", re.IGNORECASE)
_PRICE_MAX_PATTERN = re.compile(r"(?:max|budget|up to|below)\s*\$?\s*(\d+)", re.IGNORECASE)
_PRICE_EXACT_PATTERN = re.compile(r"\$(\d+)", re.IGNORECASE)

_DISPLAY_KEYWORDS = {
    "oled": "OLED",
    "amoled": "AMOLED",
    "ltpo": "LTPO AMOLED",
    "ltpo amoled": "LTPO AMOLED",
    "super amoled": "Super AMOLED",
    "ips": "IPS",
    "ips lcd": "IPS LCD",
    "lcd": "LCD",
    "retina": "Liquid Retina XDR",
    "liquid retina": "Liquid Retina XDR",
    "mini-led": "Mini-LED",
    "mini led": "Mini-LED",
}

_PROC_KEYWORDS = [
    "intel", "amd", "apple", "snapdragon", "dimensity", "tensor",
    "m3", "m4", "ultra", "npu", "ai chip", "exynos",
]


def parse_spec_query(query: str) -> ParsedSpecs:
    """
    Parse a free-text specification requirement string into structured ParsedSpecs.

    Example inputs:
      "32GB RAM OLED 144Hz"
      "16GB IPS 1TB SSD under $1500"
      "AMOLED 120Hz 5000mAh"
    """
    specs = ParsedSpecs()
    text = query.lower()

    # RAM
    m = _RAM_PATTERN.search(text)
    if m:
        specs.ram_gb = float(m.group(1))

    # Storage (TB first, then GB)
    m = _STORAGE_TB_PATTERN.search(text)
    if m:
        specs.storage_gb = float(m.group(1)) * 1024
    else:
        m = _STORAGE_GB_PATTERN.search(text)
        if m:
            specs.storage_gb = float(m.group(1))

    # Refresh rate
    m = _REFRESH_PATTERN.search(text)
    if m:
        specs.refresh_rate_hz = int(m.group(1))

    # Battery
    m = _BATTERY_PATTERN.search(text)
    if m:
        specs.battery_min_mah = int(m.group(1))

    # Camera
    m = _CAMERA_PATTERN.search(text)
    if m:
        specs.camera_min_mp = int(m.group(1))

    # Price
    for pattern in [_PRICE_UNDER_PATTERN, _PRICE_MAX_PATTERN]:
        m = pattern.search(text)
        if m:
            specs.price_max_usd = float(m.group(1))
            break
    if specs.price_max_usd is None:
        m = _PRICE_EXACT_PATTERN.search(text)
        if m:
            specs.price_max_usd = float(m.group(1)) * 1.10  # ±10% tolerance

    # Display type (check longest match first)
    for kw in sorted(_DISPLAY_KEYWORDS.keys(), key=len, reverse=True):
        if kw in text:
            specs.display_type = _DISPLAY_KEYWORDS[kw]
            break

    # Processor keywords
    specs.processor_keywords = [kw for kw in _PROC_KEYWORDS if kw in text]

    return specs


def parse_product_specs(product: dict) -> ParsedSpecs:
    """Convert a product dict (from DB row) into ParsedSpecs."""
    return ParsedSpecs(
        ram_gb=product.get("ram_gb"),
        storage_gb=product.get("storage_gb"),
        display_type=product.get("display_type"),
        refresh_rate_hz=product.get("refresh_rate_hz"),
        price_max_usd=product.get("price_usd"),
        battery_min_mah=product.get("battery_mah"),
        camera_min_mp=product.get("camera_mp"),
        processor_keywords=_extract_proc_keywords(
            str(product.get("processor") or product.get("chipset") or "")
        ),
    )


def _extract_proc_keywords(proc_str: str) -> list[str]:
    proc_lower = proc_str.lower()
    return [kw for kw in _PROC_KEYWORDS if kw in proc_lower]


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_spec_match_score(requirement: ParsedSpecs, candidate: ParsedSpecs) -> dict:
    """
    Compute a normalized [0, 1] specification match score.

    Returns a dict with:
      - overall_score  : weighted average [0, 1]
      - attribute_scores : per-attribute breakdown
      - matched_attributes : list of attribute names with score > 0
    """
    attribute_scores: dict[str, float] = {}

    # --- RAM ---
    if requirement.ram_gb is not None and candidate.ram_gb is not None:
        req_ram = requirement.ram_gb
        cand_ram = candidate.ram_gb
        if cand_ram >= req_ram:
            attribute_scores["ram"] = 1.0
        elif cand_ram >= req_ram * 0.75:
            attribute_scores["ram"] = 0.5
        else:
            attribute_scores["ram"] = 0.0

    # --- Storage ---
    if requirement.storage_gb is not None and candidate.storage_gb is not None:
        req_st = requirement.storage_gb
        cand_st = candidate.storage_gb
        if cand_st >= req_st:
            attribute_scores["storage"] = 1.0
        elif cand_st >= req_st * 0.75:
            attribute_scores["storage"] = 0.5
        else:
            attribute_scores["storage"] = 0.0

    # --- Display type ---
    if requirement.display_type is not None and candidate.display_type is not None:
        req_disp = requirement.display_type.upper()
        cand_disp = candidate.display_type.upper()
        if req_disp == cand_disp:
            attribute_scores["display"] = 1.0
        elif req_disp in cand_disp or cand_disp in req_disp:
            attribute_scores["display"] = 0.75
        elif req_disp in ("OLED", "AMOLED") and cand_disp in ("OLED", "AMOLED", "LTPO AMOLED", "SUPER AMOLED"):
            attribute_scores["display"] = 0.8
        else:
            attribute_scores["display"] = 0.0

    # --- Refresh rate ---
    if requirement.refresh_rate_hz is not None and candidate.refresh_rate_hz is not None:
        req_hz = requirement.refresh_rate_hz
        cand_hz = candidate.refresh_rate_hz
        if cand_hz >= req_hz:
            attribute_scores["refresh_rate"] = 1.0
        elif cand_hz >= req_hz * 0.80:
            attribute_scores["refresh_rate"] = 0.5
        else:
            attribute_scores["refresh_rate"] = 0.0

    # --- Price ---
    if requirement.price_max_usd is not None and candidate.price_max_usd is not None:
        if candidate.price_max_usd <= requirement.price_max_usd:
            attribute_scores["price"] = 1.0
        elif candidate.price_max_usd <= requirement.price_max_usd * 1.10:
            attribute_scores["price"] = 0.5
        else:
            attribute_scores["price"] = 0.0

    # --- Battery ---
    if requirement.battery_min_mah is not None and candidate.battery_min_mah is not None:
        if candidate.battery_min_mah >= requirement.battery_min_mah:
            attribute_scores["battery"] = 1.0
        elif candidate.battery_min_mah >= requirement.battery_min_mah * 0.85:
            attribute_scores["battery"] = 0.5
        else:
            attribute_scores["battery"] = 0.0

    # --- Camera ---
    if requirement.camera_min_mp is not None and candidate.camera_min_mp is not None:
        if candidate.camera_min_mp >= requirement.camera_min_mp:
            attribute_scores["camera"] = 1.0
        elif candidate.camera_min_mp >= requirement.camera_min_mp * 0.80:
            attribute_scores["camera"] = 0.5
        else:
            attribute_scores["camera"] = 0.0

    # --- Processor keywords ---
    if requirement.processor_keywords and candidate.processor_keywords:
        req_set = set(requirement.processor_keywords)
        cand_set = set(candidate.processor_keywords)
        overlap = len(req_set & cand_set) / len(req_set)
        attribute_scores["processor"] = overlap

    if not attribute_scores:
        return {
            "overall_score": 0.0,
            "attribute_scores": {},
            "matched_attributes": [],
        }

    overall = float(sum(attribute_scores.values()) / len(attribute_scores))
    matched = [attr for attr, score in attribute_scores.items() if score > 0]

    return {
        "overall_score": round(overall, 4),
        "attribute_scores": {k: round(v, 4) for k, v in attribute_scores.items()},
        "matched_attributes": matched,
    }


def score_product_against_query(product: dict, query: str) -> dict:
    """
    One-shot: parse a free-text query and score a product against it.
    Returns the full spec match result dict.
    """
    if not query or not query.strip():
        return {"overall_score": 0.0, "attribute_scores": {}, "matched_attributes": []}

    req = parse_spec_query(query)
    if not req.has_requirements:
        return {"overall_score": 0.0, "attribute_scores": {}, "matched_attributes": []}

    cand = parse_product_specs(product)
    return compute_spec_match_score(req, cand)
