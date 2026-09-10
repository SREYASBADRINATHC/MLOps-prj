"""
CartSense — Specification Matcher Tests
Tests the real spec parsing and scoring logic.
"""
import pytest
from backend.spec_matcher import (
    ParsedSpecs,
    parse_spec_query,
    parse_product_specs,
    compute_spec_match_score,
    score_product_against_query,
)


class TestParseSpecQuery:
    def test_ram_parsing(self):
        specs = parse_spec_query("32GB RAM")
        assert specs.ram_gb == 32.0

    def test_display_oled(self):
        specs = parse_spec_query("OLED display")
        assert specs.display_type == "OLED"

    def test_refresh_rate(self):
        specs = parse_spec_query("144Hz gaming laptop")
        assert specs.refresh_rate_hz == 144

    def test_storage_gb(self):
        specs = parse_spec_query("1TB SSD")
        assert specs.storage_gb == 1024.0

    def test_storage_gb_explicit(self):
        specs = parse_spec_query("512GB SSD")
        assert specs.storage_gb == 512.0

    def test_price_under(self):
        specs = parse_spec_query("under $1500")
        assert specs.price_max_usd == 1500.0

    def test_battery(self):
        specs = parse_spec_query("5000mAh battery")
        assert specs.battery_min_mah == 5000

    def test_camera(self):
        specs = parse_spec_query("200MP camera")
        assert specs.camera_min_mp == 200

    def test_combined(self):
        specs = parse_spec_query("32GB RAM OLED 144Hz under $2000")
        assert specs.ram_gb == 32.0
        assert specs.display_type == "OLED"
        assert specs.refresh_rate_hz == 144
        assert specs.price_max_usd == 2000.0

    def test_empty_query(self):
        specs = parse_spec_query("")
        assert not specs.has_requirements

    def test_amoled(self):
        specs = parse_spec_query("AMOLED 120Hz")
        assert specs.display_type == "AMOLED"


class TestComputeSpecMatchScore:
    def _make_product(self, ram=16, display="IPS", refresh=60, storage=512, price=999):
        return {
            "ram_gb": float(ram),
            "display_type": display,
            "refresh_rate_hz": refresh,
            "storage_gb": float(storage),
            "price_usd": float(price),
            "battery_mah": None,
            "camera_mp": None,
            "processor": None,
            "chipset": None,
        }

    def test_perfect_match(self):
        req = parse_spec_query("16GB RAM IPS 60Hz")
        cand = parse_product_specs(self._make_product(16, "IPS", 60))
        result = compute_spec_match_score(req, cand)
        assert result["overall_score"] == 1.0

    def test_partial_ram(self):
        req = parse_spec_query("32GB RAM")
        cand = parse_product_specs(self._make_product(ram=24))  # 24 >= 32*0.75 = 24 → partial
        result = compute_spec_match_score(req, cand)
        assert result["attribute_scores"]["ram"] == 0.5

    def test_insufficient_ram(self):
        req = parse_spec_query("32GB RAM")
        cand = parse_product_specs(self._make_product(ram=8))  # below 75% threshold
        result = compute_spec_match_score(req, cand)
        assert result["attribute_scores"]["ram"] == 0.0

    def test_oled_vs_ips_mismatch(self):
        req = parse_spec_query("OLED")
        cand = parse_product_specs(self._make_product(display="IPS"))
        result = compute_spec_match_score(req, cand)
        assert result["attribute_scores"]["display"] == 0.0

    def test_oled_match(self):
        req = parse_spec_query("OLED")
        cand = parse_product_specs(self._make_product(display="OLED"))
        result = compute_spec_match_score(req, cand)
        assert result["attribute_scores"]["display"] == 1.0

    def test_no_requirements_returns_zero(self):
        result = score_product_against_query({}, "")
        assert result["overall_score"] == 0.0

    def test_matched_attributes_populated(self):
        req = parse_spec_query("16GB RAM OLED 120Hz")
        cand = parse_product_specs(self._make_product(ram=16, display="OLED", refresh=120))
        result = compute_spec_match_score(req, cand)
        assert "ram" in result["matched_attributes"]
        assert "display" in result["matched_attributes"]
        assert "refresh_rate" in result["matched_attributes"]
