"""
SocioBench Data Utilities

Loads SocioBench ISSP data and maps respondent attributes to the same
8-dimension feature space used by the current WVS task/knowledge runs:
  gender, age_group, country, religion, education, marital_status,
  employment, urban_rural.
"""

import os
import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import OrderedDict

# ======================================================================
# SocioBench Domain Configuration
# ======================================================================

SOCIOBENCH_DOMAINS = OrderedDict([
    ("citizenship",       {"domain_id": 1,  "qa_file": "issp_qa_citizenship.json",       "gt_file": "issp_answer_citizenship.json"}),
    ("environment",       {"domain_id": 2,  "qa_file": "issp_qa_environment.json",       "gt_file": "issp_answer_environment.json"}),
    ("family",            {"domain_id": 3,  "qa_file": "issp_qa_family.json",            "gt_file": "issp_answer_family.json"}),
    ("health",            {"domain_id": 4,  "qa_file": "issp_qa_health.json",            "gt_file": "issp_answer_health.json"}),
    ("nationalidentity",  {"domain_id": 6,  "qa_file": "issp_qa_nationalidentity.json",  "gt_file": "issp_answer_nationalidentity.json"}),
    ("religion",          {"domain_id": 7,  "qa_file": "issp_qa_religion.json",          "gt_file": "issp_answer_religion.json"}),
    ("roleofgovernment",  {"domain_id": 8,  "qa_file": "issp_qa_roleofgovernment.json",  "gt_file": "issp_answer_roleofgovernment.json"}),
    ("socialinequality",  {"domain_id": 9,  "qa_file": "issp_qa_socialinequality.json",  "gt_file": "issp_answer_socialinequality.json"}),
    ("socialnetworks",    {"domain_id": 10, "qa_file": "issp_qa_socialnetworks.json",     "gt_file": "issp_answer_socialnetworks.json"}),
    ("workorientations",  {"domain_id": 11, "qa_file": "issp_qa_workorientations.json",   "gt_file": "issp_answer_workorientations.json"}),
])

# Country attribute key varies across domains
COUNTRY_ATTR_KEYS = {
    1:  "Country Prefix ISO 3166",
    2:  "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
    3:  "Country Prefix ISO 3166 Code - alphanumeric",
    4:  "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
    6:  "Country/ Sample Prefix ISO 3166 code - alphanumeric",
    7:  "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
    8:  "Country Prefix ISO 3166 Code - alphanumeric",
    9:  "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
    10: "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
    11: "Country Prefix ISO 3166",
}

# Map country names (as they appear in SocioBench attributes) to region
COUNTRY_NAME_TO_REGION = {
    # Asia
    "Japan": "asia", "Korea (South)": "asia", "South Korea": "asia",
    "China": "asia", "Taiwan, China": "asia", "India": "asia",
    "Philippines": "asia", "Thailand": "asia", "Israel": "asia",
    "Turkey": "asia",
    # Europe
    "Austria": "europe", "Belgium": "europe",
    "Switzerland": "europe",
    "Czech Republic": "europe", "Czechia": "europe",
    "Germany": "europe", "Germany (East)": "europe", "Germany (West)": "europe",
    "Denmark": "europe", "Spain": "europe", "Finland": "europe",
    "France": "europe", "Great Britain": "europe",
    "United Kingdom – Great Britain": "europe",
    "Georgia": "europe", "Croatia": "europe", "Hungary": "europe",
    "Iceland": "europe", "Ireland": "europe", "Italy": "europe",
    "Lithuania": "europe", "Latvia": "europe", "Estonia": "europe",
    "Netherlands": "europe", "Norway": "europe", "Poland": "europe",
    "Portugal": "europe", "Russia": "europe", "Sweden": "europe",
    "Slovenia": "europe", "Slovakia": "europe", "Slovak Republic": "europe",
    "Bulgaria": "europe",
    "Belgium–Brussels-Capital Region": "europe",
    "Belgium–Flanders": "europe", "Belgium–Wallonia": "europe",
    "Israel – Arabs": "asia", "Israel – Jews": "asia",
    # North America
    "United States of America": "north_america", "United States": "north_america",
    "United Stated": "north_america",  # typo in original data
    "Canada": "north_america", "Mexico": "north_america",
    # South America
    "Argentina": "south_america", "Venezuela": "south_america",
    "Suriname": "south_america", "Chile": "south_america",
    # Africa
    "South Africa": "africa",
    # Oceania
    "Australia": "oceania", "New Zealand": "oceania",
}

COUNTRY_NAME_TO_CODE = {
    "Argentina": "AR", "Australia": "AU", "Austria": "AT",
    "Belgium": "BE", "Belgium–Brussels-Capital Region": "BE_BRU",
    "Belgium–Flanders": "BE_FLA", "Belgium–Wallonia": "BE_WAL",
    "Bulgaria": "BG", "Canada": "CA", "Chile": "CL", "China": "CN",
    "Croatia": "HR", "Czech Republic": "CZ", "Czechia": "CZ",
    "Denmark": "DK", "Estonia": "EE", "Finland": "FI",
    "France": "FR", "Georgia": "GE", "Germany": "DE",
    "Germany (East)": "DE_E", "Germany (West)": "DE_W",
    "Great Britain": "GB_GBN",
    "United Kingdom – Great Britain": "GB_GBN",
    "Hungary": "HU", "Iceland": "IS", "India": "IN",
    "Ireland": "IE", "Israel": "IL", "Israel – Arabs": "IL_A",
    "Israel – Jews": "IL_J", "Italy": "IT", "Japan": "JP",
    "Korea (South)": "KR", "South Korea": "KR", "Latvia": "LV",
    "Lithuania": "LT", "Mexico": "MX", "Netherlands": "NL",
    "New Zealand": "NZ", "Norway": "NO", "Philippines": "PH",
    "Poland": "PL", "Portugal": "PT", "Russia": "RU",
    "Slovakia": "SK", "Slovak Republic": "SK", "Slovenia": "SI",
    "South Africa": "ZA", "Spain": "ES", "Suriname": "SR",
    "Sweden": "SE", "Switzerland": "CH", "Taiwan, China": "TW",
    "Thailand": "TH", "Turkey": "TR",
    "United Stated": "US", "United States": "US",
    "United States of America": "US", "Venezuela": "VE",
}

DEFAULT_FEATURE_DIMENSIONS = [
    "gender", "age_group", "country", "religion", "education",
    "marital_status", "employment", "urban_rural",
]

INVALID_ATTRIBUTE_MARKERS = (
    "no answer", "not available", "not applicable", "nap", "nav",
    "refused", "don't know", "dk", "other countries",
)


# ======================================================================
# Feature Extraction (SocioBench -> WVS feature space)
# ======================================================================

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in INVALID_ATTRIBUTE_MARKERS):
        return ""
    return text


def _safe_feature_value(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _first_attr(attributes: dict, keys: List[str], *, prefix: str = "") -> str:
    if prefix:
        for key, value in attributes.items():
            if prefix in key:
                cleaned = _clean_text(value)
                if cleaned:
                    return cleaned
    for key in keys:
        cleaned = _clean_text(attributes.get(key))
        if cleaned:
            return cleaned
    return ""


def _persona_first_attr(attributes: dict, keys: List[str], *,
                        prefix: str = "") -> str:
    return _first_attr(attributes, keys, prefix=prefix)


def generate_sociobench_persona_prompt(
    attributes: dict,
    domain_id: int,
    dimensions: Optional[List[str]] = None,
) -> str:
    """Build an exp2/exp4 persona from original SocioBench attribute text.

    The LoRA feature ids still use WVS-style bins such as edu_6/rel_1, but
    prompt-only evaluation should expose human-readable respondent attributes
    from SocioBench rather than those internal adapter names.
    """
    dimensions = set(dimensions or DEFAULT_FEATURE_DIMENSIONS)
    parts: List[str] = []

    if "gender" in dimensions:
        val = _persona_first_attr(
            attributes, ["Sex of Respondent", "Sex of respondent"])
        if val:
            parts.append(f"Gender: {val}")

    if "age_group" in dimensions or "age" in dimensions:
        val = _persona_first_attr(attributes, ["Age of respondent"])
        if val:
            parts.append(f"Age: {val}")

    if "country" in dimensions:
        country_key = COUNTRY_ATTR_KEYS.get(domain_id, "")
        val = _clean_text(attributes.get(country_key))
        if val:
            parts.append(f"Country: {val}")

    if "education" in dimensions:
        val = _persona_first_attr(
            attributes,
            [
                "ISCED 2011: highest completed degree of education [merged variable]",
                "ISCED 2011 simplified: highest completed degree of education",
                "Highest completed education level: Categories for international comparison",
                "Comparative - Highest completed degree of education: Categories for international comparison",
                "Education I: years of schooling",
                "Years of full-time schooling",
            ],
            prefix="Country specific highest completed degree of education",
        )
        if val:
            label = "Education"
            if re.fullmatch(r"-?\d+(?:\.\d+)?", val):
                val = f"{val} years of schooling"
            parts.append(f"{label}: {val}")

    if "religion" in dimensions:
        val = _persona_first_attr(
            attributes,
            [
                "Groups of religious affiliations (derived from nat_RELIG)",
                "Comparative: groups of religious affiliations (derived from nat_RELIG)",
            ],
            prefix="Country specific religious affiliation or denomination",
        )
        if val:
            parts.append(f"Religion: {val}")

    if "marital_status" in dimensions or "marital" in dimensions:
        val = _persona_first_attr(
            attributes,
            ["Legal partnership status", "Living in steady partnership",
             "Marital status"],
        )
        if val:
            parts.append(f"Marital status: {val}")

    if "employment" in dimensions:
        status = _persona_first_attr(
            attributes,
            ["Main status", "Currently, formerly, or never in paid work",
             "Employment relationship"],
        )
        hours = _clean_text(attributes.get("Hours worked weekly"))
        if status and hours:
            parts.append(f"Employment status: {status}; weekly work hours: {hours}")
        elif status:
            parts.append(f"Employment status: {status}")

    if "urban_rural" in dimensions:
        val = _persona_first_attr(attributes, ["Place of living: urban - rural"])
        if val:
            parts.append(f"Place of living: {val}")

    return ", ".join(parts)


def _gender_feature(attributes: dict) -> Optional[str]:
    sex_val = _first_attr(attributes, ["Sex of Respondent", "Sex of respondent"])
    lowered = sex_val.lower()
    if lowered in ("male", "1"):
        return "male"
    if lowered in ("female", "2"):
        return "female"
    return None


def _age_feature(attributes: dict) -> Optional[str]:
    age_val = _clean_text(attributes.get("Age of respondent"))
    if not age_val:
        return None
    try:
        age = int(float(age_val))
    except (ValueError, TypeError):
        return None
    if age <= 30:
        return "age_young"
    if age <= 55:
        return "age_middle"
    return "age_senior"


def _country_feature(attributes: dict, domain_id: int) -> Optional[str]:
    country_key = COUNTRY_ATTR_KEYS.get(domain_id, "")
    country_name = _clean_text(attributes.get(country_key))
    if not country_name:
        return None
    code = COUNTRY_NAME_TO_CODE.get(country_name)
    if not code:
        code = _safe_feature_value(country_name).upper()
    return f"country_{code}"


def _education_feature(attributes: dict) -> Optional[str]:
    years = _first_attr(
        attributes,
        ["Education I: years of schooling", "Years of full-time schooling"],
    )
    if years:
        try:
            match = re.search(r"-?\d+(?:\.\d+)?", years)
            y = float(match.group(0) if match else years)
            if y <= 0:
                return "edu_0"
            if y <= 6:
                return "edu_1"
            if y <= 9:
                return "edu_2"
            if y <= 12:
                return "edu_3"
            if y <= 14:
                return "edu_4"
            if y <= 16:
                return "edu_5"
            if y <= 18:
                return "edu_6"
            if y <= 20:
                return "edu_7"
            return "edu_8"
        except (ValueError, TypeError):
            pass

    text = _first_attr(
        attributes,
        [
            "Highest completed education level: Categories for international comparison",
            "Comparative - Highest completed degree of education: Categories for international comparison",
        ],
    ).lower()
    if not text:
        return None
    if "doctoral" in text:
        return "edu_8"
    if "master" in text:
        return "edu_7"
    if "bachelor" in text or "university" in text:
        return "edu_6"
    if "tertiary" in text:
        return "edu_5"
    if "post-secondary" in text:
        return "edu_4"
    if "upper secondary" in text:
        return "edu_3"
    if "lower secondary" in text:
        return "edu_2"
    if "primary" in text:
        return "edu_1"
    if "no formal" in text or "no education" in text:
        return "edu_0"
    return None


def _marital_feature(attributes: dict) -> Optional[str]:
    text = _first_attr(attributes, [
        "Legal partnership status",
        "Living in steady partnership",
        "Marital status",
    ]).lower()
    if not text:
        return None
    if "widow" in text:
        return "marital_widowed"
    if "divorc" in text:
        return "marital_divorced"
    if "separat" in text:
        return "marital_separated"
    if (
        "never married" in text
        or "never in a civil partnership" in text
        or "single" in text
        or text == "no"
    ):
        return "marital_single"
    if "married" in text or "civil partnership" in text:
        return "marital_married"
    if "steady partnership" in text or "living as couple" in text:
        return "marital_living_together"
    if text == "yes":
        return "marital_living_together"
    return None


def _religion_feature(attributes: dict) -> Optional[str]:
    text = _first_attr(
        attributes,
        ["Groups of religious affiliations (derived from nat_RELIG)"],
        prefix="Country specific religious affiliation or denomination",
    ).lower()
    if not text:
        return None
    if "no religion" in text or "no religious" in text or "none" in text:
        return "rel_0"
    if "catholic" in text:
        return "rel_1"
    if "protestant" in text or "anglican" in text:
        return "rel_2"
    if "orthodox" in text:
        return "rel_3"
    if "jew" in text:
        return "rel_4"
    if "muslim" in text or "islam" in text:
        return "rel_5"
    if "hindu" in text:
        return "rel_6"
    if "buddh" in text:
        return "rel_7"
    if "christian" in text or "relig" in text or "other" in text:
        return "rel_8"
    return None


def _employment_feature(attributes: dict) -> Optional[str]:
    main = _first_attr(attributes, [
        "Main status",
        "Currently, formerly, or never in paid work",
        "Employment relationship",
    ]).lower()
    hours = _clean_text(attributes.get("Hours worked weekly"))
    if "self-employed" in main or "self employed" in main:
        return "employment_self_employed"
    if "retired" in main:
        return "employment_retired"
    if "housework" in main or "homemaker" in main:
        return "employment_homemaker"
    if "student" in main or "education" in main:
        return "employment_student"
    if "unemployed" in main or "looking for a job" in main:
        return "employment_unemployed"
    if "paid work" in main or "employee" in main or "employed" in main:
        try:
            h = float(hours)
            return "employment_full_time" if h >= 30 else "employment_part_time"
        except (ValueError, TypeError):
            return "employment_full_time"
    if main:
        return "employment_other"
    return None


def _urban_rural_feature(attributes: dict) -> Optional[str]:
    text = _first_attr(attributes, ["Place of living: urban - rural"]).lower()
    if not text:
        return None
    if "rural" in text or "village" in text or "farm" in text:
        return "rural"
    if "urban" in text or "city" in text or "town" in text or "suburb" in text:
        return "urban"
    return None


DIMENSION_EXTRACTORS = {
    "gender": lambda attrs, did: _gender_feature(attrs),
    "age": lambda attrs, did: _age_feature(attrs),
    "age_group": lambda attrs, did: _age_feature(attrs),
    "country": lambda attrs, did: _country_feature(attrs, did),
    "education": lambda attrs, did: _education_feature(attrs),
    "marital_status": lambda attrs, did: _marital_feature(attrs),
    "marital": lambda attrs, did: _marital_feature(attrs),
    "religion": lambda attrs, did: _religion_feature(attrs),
    "employment": lambda attrs, did: _employment_feature(attrs),
    "urban_rural": lambda attrs, did: _urban_rural_feature(attrs),
    "region": lambda attrs, did: _region_feature(attrs, did),
}


def _region_feature(attributes: dict, domain_id: int) -> Optional[str]:
    country_key = COUNTRY_ATTR_KEYS.get(domain_id, "")
    country_name = _clean_text(attributes.get(country_key))
    if not country_name:
        return None
    return COUNTRY_NAME_TO_REGION.get(country_name)


def get_sociobench_user_features(
    attributes: dict,
    domain_id: int,
    dimensions: Optional[List[str]] = None,
) -> frozenset:
    """Extract demographic features from a SocioBench respondent's attributes.

    By default this returns one feature for each of the 8 WVS-style
    dimensions used in the current task/knowledge experiments.
    """
    dimensions = list(dimensions or DEFAULT_FEATURE_DIMENSIONS)
    features: Set[str] = set()
    for dim in dimensions:
        extractor = DIMENSION_EXTRACTORS.get(dim)
        if extractor is None:
            continue
        feature = extractor(attributes, domain_id)
        if feature:
            features.add(feature)
    return frozenset(features)


# ======================================================================
# Data Loading
# ======================================================================

def load_sociobench_qa(sociobench_root: str, domain_name: str) -> List[dict]:
    """Load QA file for a SocioBench domain."""
    info = SOCIOBENCH_DOMAINS[domain_name]
    path = os.path.join(sociobench_root, "Dataset_all", "q&a", info["qa_file"])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sociobench_ground_truth(sociobench_root: str, domain_name: str,
                                  dataset_size: int = 500) -> List[dict]:
    """Load ground truth respondents for a SocioBench domain."""
    info = SOCIOBENCH_DOMAINS[domain_name]
    gt_dir = f"A_GroundTruth_sampling{dataset_size}"
    path = os.path.join(sociobench_root, "Dataset_all", gt_dir, info["gt_file"])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_special_options(question_data: dict, country_code: str) -> dict:
    """Get options for a question, applying country-specific overrides."""
    options = dict(question_data.get("answer", {}))
    special = question_data.get("special", {})
    if isinstance(special, dict) and country_code and country_code in special:
        country_specific = special[country_code]
        if isinstance(country_specific, dict):
            options.update(country_specific)
    return options


def get_country_code_from_name(country_name: str, domain_id: int) -> str:
    """Reverse-lookup: country full name -> 2-letter ISO code used in SocioBench."""
    # Import the mapping from run_evaluation
    from SocioBench.evaluation.run_evaluation import COUNTRY_MAPPING
    if domain_id not in COUNTRY_MAPPING:
        return ""
    for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
        if name == country_name or name.lower() == country_name.lower():
            return code
    return ""


def is_invalid_answer(answer) -> bool:
    """Check if an answer is invalid (No answer, NAP, etc.)."""
    if answer is None:
        return True
    answer_str = str(answer).lower()
    invalid_strings = [
        "no answer", "other countries", "not available",
        "not applicable", "nap", "nav", "refused"
    ]
    return any(s in answer_str for s in invalid_strings)


def get_question_country_code(question_id: str) -> Optional[str]:
    """Extract country code from question ID like 'AT_V5' -> 'AT'."""
    m = re.match(r'^([A-Za-z\-]+)_[Vv]\d+[a-zA-Z]*$', question_id)
    return m.group(1) if m else None
