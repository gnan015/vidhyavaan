"""
Kaushal Vaani (కౌశల్ వాణి / कौशल वाणी) - Qualifications & Livelihoods Data Ingestion Pipeline.

Uses a pure-Python JSON-based vector store (data/vector_store.json).
Automatically loads on server startup via auto_initialize_vector_db().

Reads data/Qualifications.xlsx (1,581 NSQF qualification rows), builds rich text chunks,
extracts searchable trade_keywords metadata tags, and stores them in data/vector_store.json.
Provides keyword-to-sector boosting to ensure vocational trade queries (tailoring, farming, etc.)
always retrieve appropriate trades and prevent irrelevant tech recommendations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
QUALIFICATIONS_PATH = DATA_DIR / "Qualifications.xlsx"
SCHEMES_PATH = DATA_DIR / "schemes.json"
CENTERS_PATH = DATA_DIR / "centers.json"

# Primary and legacy vector store file paths
PRIMARY_VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"
LEGACY_VECTOR_STORE_DIR = Path(__file__).resolve().parents[2] / "pm_ajay_db"
LEGACY_VECTOR_STORE_PATH = LEGACY_VECTOR_STORE_DIR / "documents.json"

# Stop words to ignore during query tokenization
STOP_WORDS = {
    "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "of",
    "or", "what", "how", "i", "my", "me", "am", "want", "please", "help", "are",
    "available", "courses", "course", "training", "job", "jobs", "work", "interested",
    "can", "do", "you", "tell", "about", "give", "need", "get", "like", "pass", "have"
}

# Vocational trade intent mapping across English, Hindi, and Telugu
TRADE_INTENT_RULES: dict[str, dict[str, Any]] = {
    "apparel": {
        "terms": {
            "tailor", "tailoring", "sewing", "stitch", "stitching", "clothes",
            "garment", "garments", "dressmaker", "embroidery", "apparel", "dress",
            # Hindi
            "कपड़े", "सिलाई", "दर्जी", "सिलाई-कढ़ाई", "टेलरिंग", "कपड़ा",
            # Telugu
            "టెయిలరింగ్", "కుట్టుపని", "బట్టలు", "కుట్టు", "దర్జీ", "వస్త్రాలు",
            # Tamil
            "தையல்", "ஆடை", "தையற்கலை",
            # Kannada
            "ಹೊಲಿಗೆ", "ಟೈಲರಿಂಗ್", "ಬಟ್ಟೆ",
        },
        "sectors": ["Apparel", "Textile & Handloom", "Persons with Disability"],
        "boost_keywords": ["tailor", "sewing", "garment", "stitch", "dressmaker", "apparel"],
    },
    "agriculture": {
        "terms": {
            "farm", "farmer", "farming", "agriculture", "crop", "crops",
            "dairy", "cow", "cattle", "milk", "buffalo", "poultry", "chicken",
            "goat", "sheep", "mushroom", "pisciculture", "fishery", "cultivation",
            # Hindi
            "खेती", "किसान", "कृषि", "डेयरी", "गाय", "भैंस", "मुर्गी", "पशुपालन",
            # Telugu
            "వ్యవసాయం", "రైతు", "పాడి", "ఆవులు", "కోళ్ల", "పుట్టగొడుగులు",
            # Tamil
            "விவசாயம்", "பண்ணை", "பால்பண்ணை",
            # Kannada
            "ಕೃಷಿ", "ರೈತ", "ಡೈರಿ",
        },
        "sectors": ["Agriculture", "Food Industry/Food Processing", "Persons with Disability"],
        "boost_keywords": ["farmer", "grower", "dairy", "poultry", "agriculture", "cultivation", "mushroom"],
    },
    "electrical_green": {
        "terms": {
            "electric", "electrical", "electrician", "solar", "wiring", "wireman",
            "current", "panel", "suryamitra", "photovoltaic", "lineman", "power",
            # Hindi
            "बिजली", "इलेक्ट्रीशियन", "सोलर", "वायरिंग", "करेंट",
            # Telugu
            "ఎలక్ట్రీషియన్", "కరెంట్", "వైరింగ్", "సోలార్", "విద్యుత్",
            # Tamil
            "மின்சாரம்", "எலக்ட்ரீசியன்", "சோலார்",
            # Kannada
            "ವಿದ್ಯುತ್", "ಎಲೆಕ್ಟ್ರಿಷಿಯನ್", "ಸೋಲಾರ್",
        },
        "sectors": ["Electronics & HW", "Green Jobs", "Power", "Hydrocarbon", "Persons with Disability"],
        "boost_keywords": ["electrician", "solar", "wireman", "electrical", "photovoltaic", "technician"],
    },
    "construction_carpentry_plumbing": {
        "terms": {
            "plumb", "plumber", "plumbing", "pipe", "sanitary",
            "carpent", "carpenter", "carpentry", "wood", "furniture",
            "mason", "masonry", "welder", "welding", "construction",
            # Hindi
            "प्लंबर", "बढ़ई", "नल", "लकड़ी", "वेल्डर", "मिस्त्री",
            # Telugu
            "ప్లంబర్", "వడ్రంగి", "చెక్క", "తాపీ", "వెల్డర్",
            # Tamil
            "தச்சர்", "பிளம்பர்", "வெல்டர்",
            # Kannada
            "ಬಡಗಿ", "ಪ್ಲಂಬರ್", "ವೆಲ್ಡರ್",
        },
        "sectors": ["Construction", "Furniture & Fittings", "Plumbing", "Capital Goods & Manufacturing", "Persons with Disability"],
        "boost_keywords": ["plumber", "carpenter", "welder", "mason", "pipe", "fittings"],
    },
    "automotive": {
        "terms": {
            "mechanic", "automobile", "automotive", "car", "bike", "motorcycle",
            "scooter", "driver", "tractor", "repair",
            # Hindi
            "मैकेनिक", "गाड़ी", "ड्राइवर", "ट्रैक्टर",
            # Telugu
            "మెకానిక్", "వాహనం", "డ్రైవర్", "ట్రాక్టర్",
        },
        "sectors": ["Automotive", "Transportation, Logistics & Warehousing", "Persons with Disability"],
        "boost_keywords": ["mechanic", "driver", "automotive", "technician"],
    },
    "beauty_wellness": {
        "terms": {
            "beauty", "parlour", "salon", "hair", "makeup", "beautician",
            # Hindi
            "ब्यूटी", "पार्लर", "बाल", "मेकअप",
            # Telugu
            "బ్యూటీ పార్లర్", "మేకప్",
        },
        "sectors": ["Beauty & Wellness", "Persons with Disability"],
        "boost_keywords": ["beautician", "salon", "therapist", "beauty"],
    },
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _map_education(level: Any) -> str:
    """
    Map NSQF Level to minimum education requirement:
      Level 2 -> "5th pass / Non-literate"
      Level 3 -> "8th to 10th pass"
      Level 4 -> "10th/12th/ITI"
    """
    lvl = str(level or "").strip()
    if "2" in lvl:
        return "5th pass / Non-literate"
    if "3" in lvl:
        return "8th to 10th pass"
    if "4" in lvl:
        return "10th/12th/ITI"
    return "No formal education required"


def _map_career_type(title: str) -> str:
    """
    Mark as "Self-Employment / Micro-Enterprise" if title contains
    "farmer", "grower", "maker", "artisan", "entrepreneur", "tailor", "udyami";
    otherwise "Wage Employment".
    """
    t = str(title or "").lower()
    self_keywords = ["farmer", "grower", "maker", "artisan", "entrepreneur", "tailor", "udyami"]
    if any(kw in t for kw in self_keywords):
        return "Self-Employment / Micro-Enterprise"
    return "Wage Employment"


def _map_is_pwd(sector_name: str) -> str:
    """Return "Yes" if Sector Name is "Persons with Disability", else "No"."""
    if str(sector_name or "").strip().lower() == "persons with disability":
        return "Yes"
    return "No"


def _extract_trade_keywords(title: str, sector: str, description: str) -> list[str]:
    """Generate clean, searchable trade keywords for every qualification."""
    keywords = set()
    combined = (f"{title} {sector} {description}").lower()

    trade_maps = {
        "tailor": ["tailor", "sewing", "garment", "stitch", "apparel", "fashion", "dressmaker", "embroidery"],
        "sew": ["tailor", "sewing", "garment", "stitch", "apparel"],
        "apparel": ["tailor", "sewing", "garment", "stitch", "apparel", "fashion"],
        "garment": ["garment", "tailor", "sewing", "apparel", "stitch"],
        "stitch": ["stitch", "tailor", "sewing", "garment", "apparel"],
        "farm": ["farmer", "farming", "agriculture", "crop", "cultivation", "dairy", "poultry"],
        "agriculture": ["agriculture", "farming", "crop", "farmer", "cultivation", "dairy", "poultry"],
        "dairy": ["dairy", "cattle", "milk", "farming", "cow", "buffalo", "livestock"],
        "poultry": ["poultry", "chicken", "farming", "egg", "broiler"],
        "mushroom": ["mushroom", "grower", "cultivation", "farming"],
        "electric": ["electrician", "electrical", "wiring", "power", "electronics", "technician"],
        "solar": ["solar", "photovoltaic", "suryamitra", "green jobs", "renewable", "installer", "panel"],
        "plumb": ["plumber", "plumbing", "pipe", "sanitary", "fitting"],
        "carpent": ["carpenter", "carpentry", "wood", "furniture", "fittings"],
        "weld": ["welder", "welding", "metal", "fabrication"],
        "mason": ["mason", "masonry", "construction", "building"],
        "construct": ["construction", "building", "mason", "fitter"],
        "mechanic": ["mechanic", "automotive", "automobile", "repair", "vehicle", "driver"],
        "auto": ["automotive", "vehicle", "automobile", "mechanic", "driver"],
        "beauty": ["beautician", "beauty", "hair", "salon", "makeup", "wellness"],
        "handicraft": ["handicraft", "artisan", "carpet", "weaving", "handloom", "weaver"],
        "food": ["food processing", "preservation", "baking", "pickle", "food"],
        "leather": ["leather", "footwear", "shoe"],
    }

    for trigger, kw_list in trade_maps.items():
        if trigger in combined:
            keywords.update(kw_list)

    for token in re.findall(r"[a-z0-9]+", title.lower()):
        if len(token) > 2 and token not in {"and", "for", "the", "with", "level", "assistant", "technician", "operator"}:
            keywords.add(token)

    return sorted(keywords)


def _build_qualification_chunk(row: dict[str, Any]) -> str:
    title = str(row.get("Title") or row.get("Qualification Title") or "").strip()
    sector = str(row.get("Sector Name") or "").strip()
    level = str(row.get("Level") or row.get("NSQF Level") or "").strip()
    description = str(row.get("Description") or row.get("Brief Description") or "").strip()
    progression = str(row.get("Progression Pathway") or row.get("Progression") or "").strip()
    hours = str(row.get("Training Delivery Hours") or row.get("Maximum Notational Hours") or "").strip()

    career_type = _map_career_type(title)
    min_edu = _map_education(level)
    is_pwd = _map_is_pwd(sector)

    parts = [
        f"Job Role: {title}.",
        f"Sector: {sector}.",
        f"NSQF Level: {level}. Minimum Education: {min_edu}.",
        f"Career Type: {career_type}.",
        f"Suitable for PwD: {is_pwd}.",
    ]
    if description:
        parts.append(f"Description: {description}.")
    if progression:
        parts.append(f"Progression Pathway: {progression}.")
    if hours:
        parts.append(f"Training Duration: {hours}.")
    return " ".join(parts)


def _load_qualifications_xlsx() -> list[dict[str, Any]]:
    """Load qualification rows from Excel file skipping initial title rows."""
    import openpyxl

    if not QUALIFICATIONS_PATH.is_file():
        logger.warning("qualifications_xlsx_missing path=%s", QUALIFICATIONS_PATH)
        return []

    wb = openpyxl.load_workbook(str(QUALIFICATIONS_PATH), read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        return []

    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return []

    header_idx = None
    for i, r in enumerate(rows[:10]):
        str_vals = [str(v or "").strip().lower() for v in r]
        if "title" in str_vals and any(k in str_vals for k in ("sector name", "sector", "level")):
            header_idx = i
            break

    if header_idx is None:
        header_idx = 0

    headers = [str(h or "").strip() for h in rows[header_idx]]
    records = []
    for row in rows[header_idx + 1:]:
        rec = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
        title = rec.get("Title") or rec.get("Qualification Title")
        if title and str(title).strip():
            records.append(rec)

    logger.info("qualifications_loaded rows=%d", len(records))
    return records


def _build_document_store() -> list[dict[str, Any]]:
    """Build all documents from all sources into a flat list with rich trade_keywords metadata."""
    docs: list[dict[str, Any]] = []

    # 1. Qualifications.xlsx (1,581 NSQF rows)
    records = _load_qualifications_xlsx()
    for i, row in enumerate(records):
        chunk = _build_qualification_chunk(row)
        title = str(row.get("Title") or row.get("Qualification Title") or f"role_{i}").strip()
        sector = str(row.get("Sector Name") or "").strip()
        level = str(row.get("Level") or row.get("NSQF Level") or "").strip()
        desc = str(row.get("Description") or row.get("Brief Description") or "")
        career_type = _map_career_type(title)
        min_edu = _map_education(level)
        is_pwd = _map_is_pwd(sector)
        trade_kws = _extract_trade_keywords(title, sector, desc)

        docs.append({
            "id": f"qual_{i:04d}",
            "text": chunk,
            "source": "Qualifications.xlsx",
            "title": title[:200],
            "sector": sector[:100],
            "level": level,
            "min_education": min_edu,
            "career_type": career_type,
            "is_pwd": is_pwd,
            "trade_keywords": trade_kws,
        })

    # 2. schemes.json
    if SCHEMES_PATH.is_file():
        try:
            schemes = json.loads(SCHEMES_PATH.read_text(encoding="utf-8"))
            pm = schemes.get("pm_ajay", schemes)

            docs.append({
                "id": "scheme_overview",
                "text": (
                    f"PM-AJAY stands for {pm.get('full_name', 'PM-AJAY')}. "
                    f"Ministry: {pm.get('ministry', 'MoSJE')}. "
                    f"Target: {pm.get('target_beneficiaries', '')}. "
                    f"Income ceiling: Rs {pm.get('income_eligibility_ceiling_inr_pa', 250000):,} per year. "
                    f"{pm.get('description', '')}"
                ),
                "source": "schemes.json",
                "type": "overview",
                "trade_keywords": ["pm-ajay", "subsidy", "scheme", "grant", "livelihood", "eligibility"],
            })

            for comp_key, comp in pm.get("components", {}).items():
                if isinstance(comp, dict):
                    text = f"PM-AJAY Component: {comp.get('name', comp_key)}. "
                    for k, v in comp.items():
                        if k != "name" and isinstance(v, (str, int, float, list)):
                            val = ", ".join(v) if isinstance(v, list) else str(v)
                            text += f"{k.replace('_', ' ').title()}: {val}. "
                    docs.append({
                        "id": f"comp_{comp_key}",
                        "text": text,
                        "source": "schemes.json",
                        "type": "component",
                        "trade_keywords": ["component", comp_key, "skill training", "subsidy", "infrastructure"],
                    })

            credit = pm.get("components", {}).get("credit_linkages", {})
            for key, scheme in credit.items():
                if isinstance(scheme, dict):
                    docs.append({
                        "id": f"credit_{key}",
                        "text": (
                            f"Credit Scheme: {scheme.get('name', key)}. "
                            f"Max loan: Rs {scheme.get('max_loan_inr', 'N/A')}. "
                            f"Interest rate: {scheme.get('interest_rate_percent', 'N/A')}%. "
                            f"For: {scheme.get('for', '')}."
                        ),
                        "source": "schemes.json",
                        "type": "credit",
                        "trade_keywords": ["loan", "credit", "nsfdc", "mudra", "interest", "finance"],
                    })

            for key, model in pm.get("priority_livelihood_models", {}).items():
                if isinstance(model, dict):
                    trades = model.get("trades", model.get("sub_models", []))
                    docs.append({
                        "id": f"model_{key}",
                        "text": (
                            f"PM-AJAY Priority Model: {model.get('full_name', model.get('name', key))}. "
                            f"Trades: {', '.join(trades) if trades else ''}. "
                            f"Income: {model.get('income_range_inr_pm', '')}. "
                            f"{model.get('description', '')}"
                        ),
                        "source": "schemes.json",
                        "type": "model",
                        "trade_keywords": ["livelihood", "model", key] + [t.lower() for t in trades],
                    })

            app = pm.get("application_process", {})
            if app:
                docs_req = ", ".join(app.get("documents_required", []))
                docs.append({
                    "id": "scheme_application",
                    "text": (
                        f"To apply for PM-AJAY: visit DRDA or Block Development Office. "
                        f"Online: {app.get('online', '')}. "
                        f"Documents: {docs_req}."
                    ),
                    "source": "schemes.json",
                    "type": "application",
                    "trade_keywords": ["apply", "drda", "documents", "application", "portal"],
                })
        except Exception as exc:
            logger.warning("schemes_json_parse_error err=%s", exc)

    # 3. centers.json
    if CENTERS_PATH.is_file():
        try:
            centers = json.loads(CENTERS_PATH.read_text(encoding="utf-8"))
            idx = 0
            for state, districts in centers.items():
                if not isinstance(districts, dict):
                    continue
                for district, center_list in districts.items():
                    if not isinstance(center_list, list):
                        continue
                    for center in center_list:
                        trades = ", ".join(center.get("popular_trades", []))
                        sectors = ", ".join(center.get("key_sectors", []))
                        trade_kws = [t.lower() for t in center.get("popular_trades", [])] + [s.lower() for s in center.get("key_sectors", [])]
                        docs.append({
                            "id": f"center_{idx:04d}",
                            "text": (
                                f"Skill Training Center: {center.get('center_name', '')}. "
                                f"Location: {district}, {state}. "
                                f"Type: {center.get('type', '')}. "
                                f"Address: {center.get('address', '')}. "
                                f"Phone: {center.get('phone', '')}. "
                                f"Sectors: {sectors}. "
                                f"Trades: {trades}."
                            ),
                            "source": "centers.json",
                            "type": "training_center",
                            "state": state,
                            "district": district,
                            "trade_keywords": trade_kws,
                        })
                        idx += 1
        except Exception as exc:
            logger.warning("centers_json_parse_error err=%s", exc)

    return docs


def _save_document_store(docs: list[dict[str, Any]]) -> None:
    PRIMARY_VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_primary = PRIMARY_VECTOR_STORE_PATH.with_suffix(".tmp")
    content = json.dumps(docs, ensure_ascii=False, indent=None)
    tmp_primary.write_text(content, encoding="utf-8")
    tmp_primary.replace(PRIMARY_VECTOR_STORE_PATH)

    LEGACY_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_legacy = LEGACY_VECTOR_STORE_PATH.with_suffix(".tmp")
    tmp_legacy.write_text(content, encoding="utf-8")
    tmp_legacy.replace(LEGACY_VECTOR_STORE_PATH)

    logger.info("vector_store_saved path=%s total=%d", PRIMARY_VECTOR_STORE_PATH, len(docs))


def _load_document_store() -> list[dict[str, Any]]:
    for path in (PRIMARY_VECTOR_STORE_PATH, LEGACY_VECTOR_STORE_PATH):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) >= 100:
                    return data
            except Exception as exc:
                logger.warning("vector_store_load_error path=%s err=%s", path, exc)
    return []


def _store_record_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _store_has_data() -> bool:
    return len(_load_document_store()) > 0


def _wipe_stale_cache_if_needed() -> bool:
    """
    Check document count and schema in existing vector store.
    If count < 100 or missing 'trade_keywords', wipe cache and return True (needs re-ingest).
    """
    for path in (PRIMARY_VECTOR_STORE_PATH, LEGACY_VECTOR_STORE_PATH):
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, list) or len(data) < 100:
                    _wipe_all_caches()
                    return True
                qual_sample = next((d for d in data if d.get("source") == "Qualifications.xlsx"), None)
                if qual_sample and "trade_keywords" not in qual_sample:
                    logger.info("stale_schema_detected (missing trade_keywords), wiping vector store cache")
                    _wipe_all_caches()
                    return True
            except Exception:
                _wipe_all_caches()
                return True
        else:
            return True
    return False


def _wipe_all_caches() -> None:
    if PRIMARY_VECTOR_STORE_PATH.is_file():
        PRIMARY_VECTOR_STORE_PATH.unlink(missing_ok=True)
    if LEGACY_VECTOR_STORE_PATH.is_file():
        LEGACY_VECTOR_STORE_PATH.unlink(missing_ok=True)


def _run_full_ingestion() -> int:
    """Run complete ingestion. Returns number of documents stored."""
    docs = _build_document_store()
    _save_document_store(docs)
    return len(docs)


# ─── Trade Intent & Search ───────────────────────────────────────────────────

def _detect_trade_intent(query: str) -> tuple[list[str], list[str]]:
    """Detect vocational trade intent and return (preferred_sectors, boost_keywords)."""
    q_lower = query.lower()
    q_tokens = set(re.findall(r"[\u0C00-\u0C7F]+|[\u0900-\u097F]+|[a-z0-9]+", q_lower))
    sectors: list[str] = []
    boosts: list[str] = []

    for trade_key, config in TRADE_INTENT_RULES.items():
        if q_tokens.intersection(config["terms"]) or any(t in q_lower for t in config["terms"]):
            sectors.extend(config["sectors"])
            boosts.extend(config["boost_keywords"])

    return list(dict.fromkeys(sectors)), list(dict.fromkeys(boosts))


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\u0C00-\u0C7F]+|[\u0900-\u097F]+|[a-z0-9]+", text.lower())
    return [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]


def _keyword_search(
    query: str,
    docs: list[dict[str, Any]],
    n: int = 5,
    preferred_sectors: list[str] | None = None,
    boost_keywords: list[str] | None = None,
) -> list[str]:
    """
    Search documents with strict vocational trade boosting and filtering.
    Prevents irrelevant tech recommendations when caller asks for vocational trades.
    """
    if preferred_sectors is None or boost_keywords is None:
        auto_sectors, auto_boosts = _detect_trade_intent(query)
        preferred_sectors = preferred_sectors or auto_sectors
        boost_keywords = boost_keywords or auto_boosts

    pref_sectors_lower = {s.lower() for s in (preferred_sectors or [])}
    boost_kws_lower = {b.lower() for b in (boost_keywords or [])}

    query_terms = set(_tokenize(query))

    if not query_terms and not boost_kws_lower:
        return [d["text"] for d in docs[:n]]

    scored: list[tuple[float, str]] = []
    has_trade_intent = bool(pref_sectors_lower or boost_kws_lower)

    for doc in docs:
        text = doc.get("text", "")
        title = doc.get("title", "").lower()
        sector = doc.get("sector", "").lower()
        trade_kws = [k.lower() for k in doc.get("trade_keywords", [])]
        doc_words = set(re.findall(r"[a-z0-9]+", text.lower()))

        score = 0.0

        # Term frequency matching
        for term in query_terms:
            if term in doc_words:
                score += 3.0
            if term in title:
                score += 25.0
            if any(term in kw for kw in trade_kws):
                score += 15.0

        # Vocational trade boosting
        if has_trade_intent:
            if any(s in sector for s in pref_sectors_lower):
                score += 80.0
            matching_boosts = min(3, sum(1 for b in boost_kws_lower if b in title or any(b in kw for kw in trade_kws)))
            score += matching_boosts * 35.0

            # Suppress unrelated tech courses (IT-ITeS, Telecom) for vocational queries
            if any(tech in sector for tech in ("it-ites", "information technology", "telecom")):
                score -= 150.0

        # Keep relevant scheme overviews, subsidies, and application documents accessible
        if doc.get("source") == "schemes.json":
            scheme_intent_words = {"subsidy", "asset", "50000", "50,000", "loan", "credit", "grant", "rickshaw", "drda", "income", "eligibility"}
            matched_scheme_words = sum(1 for w in scheme_intent_words if w in query_terms and (w in doc_words or any(w in kw for kw in trade_kws)))
            if matched_scheme_words > 0:
                score += 180.0 + (matched_scheme_words * 40.0)
            elif any(kw in query.lower() for kw in ("scheme", "pm-ajay", "apply", "benefit")):
                score += 40.0
        elif doc.get("source") == "centers.json":
            if any(kw in query.lower() for kw in ("center", "centre", "district", "address", "location")):
                score += 50.0

        if score > 0:
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:n]]


# ─── Cached document store ────────────────────────────────────────────────────
_CACHED_DOCS: list[dict[str, Any]] | None = None


def _get_docs() -> list[dict[str, Any]]:
    global _CACHED_DOCS
    if _CACHED_DOCS is None:
        _CACHED_DOCS = _load_document_store()
    return _CACHED_DOCS


# ─── Public API ───────────────────────────────────────────────────────────────

async def auto_initialize_vector_db() -> None:
    """
    FastAPI lifespan hook — auto-runs at server startup.
    Checks vector store document count; if < 100 or missing trade_keywords, wipes cache and ingests.
    """
    try:
        needs_ingest = await asyncio.to_thread(_wipe_stale_cache_if_needed)
        if not needs_ingest:
            await asyncio.to_thread(_get_docs)
            count = len(_get_docs())
            logger.info("[KAUSHAL VAANI] Ingestion complete. Indexed %d records into livelihood vector store.", count)
            print(f"[KAUSHAL VAANI] Ingestion complete. Indexed {count} records into livelihood vector store.", flush=True)
            return

        logger.info("kaushal_vaani_ingestion_start", extra={"event": "ingestion"})
        total = await asyncio.to_thread(_run_full_ingestion)
        global _CACHED_DOCS
        _CACHED_DOCS = None
        await asyncio.to_thread(_get_docs)
        logger.info("[KAUSHAL VAANI] Ingestion complete. Indexed %d records into livelihood vector store.", total)
        print(f"[KAUSHAL VAANI] Ingestion complete. Indexed {total} records into livelihood vector store.", flush=True)
    except Exception:
        logger.exception("kaushal_vaani_ingestion_failed", extra={"event": "ingestion"})
        print("[KAUSHAL VAANI] WARNING: Vector DB initialization failed. RAG uses Groq only.", flush=True)


async def query_vector_db(
    query: str,
    n_results: int = 5,
    filter_meta: dict | None = None,
    preferred_sectors: list[str] | None = None,
    boost_keywords: list[str] | None = None,
) -> list[str]:
    """
    Query the Kaushal Vaani JSON document store with vocational trade boosting.
    """
    def _search_sync() -> list[str]:
        docs = _get_docs()
        if filter_meta:
            docs = [d for d in docs if all(d.get(k) == v for k, v in filter_meta.items())]
        return _keyword_search(
            query,
            docs,
            n=n_results,
            preferred_sectors=preferred_sectors,
            boost_keywords=boost_keywords,
        )

    try:
        return await asyncio.to_thread(_search_sync)
    except Exception as exc:
        logger.warning("vector_db_query_failed err=%s", exc)
        return []
