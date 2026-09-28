import json
from pathlib import Path
from typing import Any

CENTERS_FILE = Path("app/data/centers.json")

def load_centers_data() -> dict[str, Any]:
    if not CENTERS_FILE.exists():
        return {}
    with open(CENTERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_recommended_center(district: str, recommended_trade: str, state: str = "Andhra Pradesh") -> dict[str, str]:
    """Return the best-matching training center for the caller's district and trade."""
    data = load_centers_data()
    district_centers = data.get(state, {}).get(district)

    # Fallback to Visakhapatnam if caller's district is not yet cataloged
    if not district_centers:
        district_centers = data.get(state, {}).get("Visakhapatnam", [])

    # Priority 1: Match center offering the specific trade
    for center in district_centers:
        for trade in center.get("popular_trades", []):
            if trade.lower() in recommended_trade.lower() or recommended_trade.lower() in trade.lower():
                return center

    # Priority 2: Match RSETI if it's an enterprise/farming trade
    trade_lower = recommended_trade.lower()
    is_self_employed = any(k in trade_lower for k in ["farmer", "tailor", "grower", "artisan", "maker"])
    for center in district_centers:
        if is_self_employed and "RSETI" in center.get("type", ""):
            return center

    # Default fallback: Return the first official Government ITI
    return district_centers[0] if district_centers else {
        "center_name": "District Skill Development Center",
        "address": f"District Collectorate Compound, {district}",
        "phone": "1800-425-2422"
    }