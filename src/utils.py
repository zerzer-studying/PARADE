import os
import json
import csv
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Set, Optional
from collections import defaultdict
import numpy as np
import random

from anonymization import public_path

try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from lora import MultiLoRAModelWrapper
from config import *

COUNTRY_NAME_BY_ALPHA3 = {
    'AND': 'Andorra',
    'ARG': 'Argentina',
    'ARM': 'Armenia',
    'AUS': 'Australia',
    'BGD': 'Bangladesh',
    'BOL': 'Bolivia',
    'BRA': 'Brazil',
    'CAN': 'Canada',
    'CHL': 'Chile',
    'CHN': 'China',
    'COL': 'Colombia',
    'CYP': 'Cyprus',
    'CZE': 'Czech Republic',
    'DEU': 'Germany',
    'ECU': 'Ecuador',
    'EGY': 'Egypt',
    'ETH': 'Ethiopia',
    'GBR': 'United Kingdom',
    'GRC': 'Greece',
    'GTM': 'Guatemala',
    'HKG': 'Hong Kong',
    'IDN': 'Indonesia',
    'IND': 'India',
    'IRN': 'Iran',
    'IRQ': 'Iraq',
    'JOR': 'Jordan',
    'JPN': 'Japan',
    'KAZ': 'Kazakhstan',
    'KEN': 'Kenya',
    'KGZ': 'Kyrgyzstan',
    'KOR': 'South Korea',
    'LBN': 'Lebanon',
    'LBY': 'Libya',
    'MAC': 'Macau',
    'MAR': 'Morocco',
    'MDV': 'Maldives',
    'MEX': 'Mexico',
    'MMR': 'Myanmar',
    'MNG': 'Mongolia',
    'MYS': 'Malaysia',
    'NGA': 'Nigeria',
    'NIC': 'Nicaragua',
    'NIR': 'Northern Ireland',
    'NLD': 'Netherlands',
    'NZL': 'New Zealand',
    'PAK': 'Pakistan',
    'PER': 'Peru',
    'PHL': 'Philippines',
    'PRI': 'Puerto Rico',
    'ROU': 'Romania',
    'RUS': 'Russia',
    'SGP': 'Singapore',
    'SRB': 'Serbia',
    'SVK': 'Slovakia',
    'THA': 'Thailand',
    'TJK': 'Tajikistan',
    'TUN': 'Tunisia',
    'TUR': 'Turkey',
    'TWN': 'Taiwan',
    'UKR': 'Ukraine',
    'URY': 'Uruguay',
    'USA': 'United States',
    'UZB': 'Uzbekistan',
    'VEN': 'Venezuela',
    'VNM': 'Vietnam',
    'ZWE': 'Zimbabwe',
}

EDUCATION_LABEL_BY_CODE = {
    '0': 'early childhood education or no education',
    '1': 'primary education',
    '2': 'lower secondary education',
    '3': 'upper secondary education',
    '4': 'post-secondary non-tertiary education',
    '5': 'short-cycle tertiary education',
    '6': "Bachelor's degree or equivalent",
    '7': "Master's degree or equivalent",
    '8': 'doctoral degree or equivalent',
}

RELIGION_LABEL_BY_CODE = {
    '0': 'no religious denomination',
    '1': 'Roman Catholic',
    '2': 'Protestant',
    '3': 'Orthodox Christian',
    '4': 'Jewish',
    '5': 'Muslim',
    '6': 'Hindu',
    '7': 'Buddhist',
    '8': 'other religious denomination',
    '9': 'other religious denomination',
}

MARITAL_LABEL_BY_FEATURE = {
    'marital_married': 'married',
    'marital_living_together': 'living together as married',
    'marital_divorced': 'divorced',
    'marital_separated': 'separated',
    'marital_widowed': 'widowed',
    'marital_single': 'single and never married',
}

EMPLOYMENT_LABEL_BY_FEATURE = {
    'employment_full_time': 'employed full-time, 30 hours a week or more',
    'employment_part_time': 'employed part-time, less than 30 hours a week',
    'employment_self_employed': 'self-employed',
    'employment_retired': 'retired or receiving a pension',
    'employment_homemaker': 'a homemaker and not otherwise employed',
    'employment_student': 'a student',
    'employment_unemployed': 'unemployed',
    'employment_other': 'in another employment situation',
}

OCCUPATION_LABEL_BY_CODE = {
    '0': 'never had a job',
    '1': 'in the professional and technical group, such as doctor, teacher, engineer, artist, accountant, or nurse',
    '2': 'in the higher administrative group, such as banker, executive in big business, high government official, or union official',
    '3': 'in the clerical group, such as secretary, clerk, office manager, civil servant, or bookkeeper',
    '4': 'in the sales group, such as sales manager, shop owner, shop assistant, insurance agent, or buyer',
    '5': 'in the service group, such as restaurant owner, police officer, waitress, barber, or caretaker',
    '6': 'in the skilled worker group, such as foreman, motor mechanic, printer, seamstress, tool and die maker, or electrician',
    '7': 'in the semi-skilled worker group, such as bricklayer, bus driver, cannery worker, carpenter, sheet metal worker, or baker',
    '8': 'in the unskilled worker group, such as laborer, porter, unskilled factory worker, or cleaner',
    '9': 'in the farm worker group, such as farm laborer or tractor driver',
    '10': 'a farm proprietor or farm manager',
    '11': 'in another or uncategorized occupational group',
}


def generate_persona_prompt(features: frozenset) -> str:
    """Generate persona prompt from demographic features as a system prompt.

    Example: "You are a person with the following profile: Gender: male, Region: Asia, Age: young (30 or under). You are a helpful assistant that answers survey questions honestly."
    """
    profile_parts = []

    # Gender
    if 'male' in features or 'gender_male' in features:
        profile_parts.append("Gender: male")
    elif 'female' in features or 'gender_female' in features:
        profile_parts.append("Gender: female")

    # Region
    region_map = {
        'asia': 'Asia',
        'europe': 'Europe',
        'north_america': 'North America',
        'south_america': 'South America',
        'africa': 'Africa',
        'oceania': 'Oceania',
    }
    for region_key, region_name in region_map.items():
        if region_key in features:
            profile_parts.append(f"Region: {region_name}")
            break

    # Age
    if 'young' in features or 'age_young' in features:
        profile_parts.append("Age: young (30 or under)")
    elif 'middle_aged' in features or 'age_middle' in features:
        profile_parts.append("Age: middle-aged (31-55)")
    elif 'old' in features or 'age_senior' in features:
        profile_parts.append("Age: older (56 or above)")

    for feature in sorted(features):
        if feature.startswith('country_'):
            country_code = feature[len('country_'):].upper()
            country_name = COUNTRY_NAME_BY_ALPHA3.get(country_code)
            if country_name:
                profile_parts.append(f"Country: {country_name}")
            else:
                profile_parts.append(
                    f"Country: {country_code.replace('_', ' ')}")
            break

    for feature in sorted(features):
        if feature.startswith('edu_') and feature not in {'edu_low', 'edu_medium', 'edu_high'}:
            edu_code = feature[len('edu_'):]
            edu_label = EDUCATION_LABEL_BY_CODE.get(edu_code)
            if edu_label:
                profile_parts.append(f"Education: {edu_label}")
            else:
                profile_parts.append(
                    f"Education: unknown level {edu_code}")
            break
    if 'edu_low' in features:
        profile_parts.append("Education: low")
    elif 'edu_medium' in features:
        profile_parts.append("Education: medium")
    elif 'edu_high' in features:
        profile_parts.append("Education: high")

    for feature in sorted(features):
        if feature.startswith('rel_'):
            rel_code = feature[len('rel_'):]
            rel_label = RELIGION_LABEL_BY_CODE.get(rel_code)
            if rel_label:
                profile_parts.append(f"Religion: {rel_label}")
            else:
                profile_parts.append(
                    f"Religion: unknown denomination {rel_code}")
            break

    for feature in sorted(features):
        if feature.startswith('marital_'):
            marital_label = MARITAL_LABEL_BY_FEATURE.get(
                feature, feature[len('marital_'):].replace('_', ' '))
            profile_parts.append(f"Marital status: {marital_label}")
            break

    for feature in sorted(features):
        if feature.startswith('eth_'):
            profile_parts.append(f"Ethnicity code: {feature[len('eth_'):]}")
            break

    for feature in sorted(features):
        if feature.startswith('employment_'):
            employment_label = EMPLOYMENT_LABEL_BY_FEATURE.get(
                feature, feature[len('employment_'):].replace('_', ' '))
            profile_parts.append(f"Employment status: {employment_label}")
            break

    for feature in sorted(features):
        if feature.startswith('occupation_'):
            occ_code = feature[len('occupation_'):]
            occupation_label = OCCUPATION_LABEL_BY_CODE.get(
                occ_code, f"occupation group {occ_code}")
            profile_parts.append(f"Occupation: {occupation_label}")
            break

    profile_str = ", ".join(profile_parts)
    return profile_str


def joint_adapter_name(features) -> str:
    """Stable adapter name for a jointly trained feature combination."""
    return "__".join(sorted(features))


def extract_answer_number(response: str, valid_options: Optional[frozenset] = None) -> Optional[int]:
    """Extract the predicted option number from generated text.

    First tries to extract from <answer>number</answer> tags.
    Falls back to other patterns if tags are not found.

    If valid_options (a set of ints) is provided, the extracted number must
    be a member; otherwise the prediction is treated as invalid (None).
    """
    def _check(n):
        if valid_options is not None and n not in valid_options:
            return None
        return n

    try:
        # Priority 1: Extract from <answer> tags
        m = re.search(r"<answer>\s*(-?\d+)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        if m:
            result = _check(int(m.group(1)))
            if result is not None:
                return result

        # Priority 2: Try to match "Answer: <number>"
        m = re.search(r"Answer:\s*(-?\d+)", response, re.IGNORECASE)
        if m:
            result = _check(int(m.group(1)))
            if result is not None:
                return result

        # Fallback: match number at the beginning
        m = re.search(r"^\s*(-?\d+)\b", response)
        if m:
            return _check(int(m.group(1)))
    except Exception as e:
        print(f"Warning: Error extracting answer from response: {e}")

    return None

def _clean_feature_value(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if float(s) < 0:
            return None
    except ValueError:
        pass
    return s


def _safe_feature_value(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")


def _education_feature(row: dict, *, raw: bool = False) -> Optional[str]:
    edu_str = _clean_feature_value(row.get('Q275'))
    if edu_str is None:
        return None
    try:
        edu = int(float(edu_str))
    except (ValueError, TypeError):
        return None
    if raw:
        return f"edu_{edu}" if 0 <= edu <= 8 else None
    if 0 <= edu <= 2:
        return 'edu_low'
    if 3 <= edu <= 5:
        return 'edu_medium'
    if 6 <= edu <= 8:
        return 'edu_high'
    return None


def _age_group_feature(row: dict, *, legacy: bool = False) -> Optional[str]:
    age_str = _clean_feature_value(row.get('Q262'))
    if age_str is None:
        return None
    try:
        age = float(age_str)
    except (ValueError, TypeError):
        return None
    if legacy:
        if age <= 30:
            return 'young'
        if age <= 55:
            return 'middle_aged'
        return 'old'
    if age <= 30:
        return 'age_young'
    if age <= 55:
        return 'age_middle'
    return 'age_senior'


def _marital_feature(row: dict) -> Optional[str]:
    m_str = _clean_feature_value(row.get('Q273'))
    if m_str is None:
        return None
    try:
        m = int(float(m_str))
    except (ValueError, TypeError):
        return None
    return {
        1: 'marital_married',
        2: 'marital_living_together',
        3: 'marital_divorced',
        4: 'marital_separated',
        5: 'marital_widowed',
        6: 'marital_single',
    }.get(m)


def _employment_feature(row: dict) -> Optional[str]:
    emp_str = _clean_feature_value(row.get('Q279'))
    if emp_str is None:
        return None
    try:
        emp = int(float(emp_str))
    except (ValueError, TypeError):
        return None
    return {
        1: 'employment_full_time',
        2: 'employment_part_time',
        3: 'employment_self_employed',
        4: 'employment_retired',
        5: 'employment_homemaker',
        6: 'employment_student',
        7: 'employment_unemployed',
        8: 'employment_other',
    }.get(emp)


def _occupation_feature(row: dict) -> Optional[str]:
    occ_str = _clean_feature_value(row.get('Q281'))
    if occ_str is None:
        return None
    try:
        occ = int(float(occ_str))
    except (ValueError, TypeError):
        return None
    return f'occupation_{occ}' if 0 <= occ <= 11 else None


def _religion_feature(row: dict) -> Optional[str]:
    religion = _clean_feature_value(row.get('Q289'))
    if religion is None:
        return None
    return f"rel_{_safe_feature_value(religion)}"


def _ethnicity_feature(row: dict) -> Optional[str]:
    ethnicity = _clean_feature_value(row.get('Q290'))
    if ethnicity is None:
        return None
    return f"eth_{_safe_feature_value(ethnicity)}"


def parse_feature_group_from_filename(filename: str) -> Optional[frozenset]:
    """Parse a *_train.json filename into feature adapter names."""
    demographics_str = filename.replace('_train.json', '')

    v2_match = re.fullmatch(
        r"country_(?P<country>.+?)_edu_(?P<edu>-?\d+)_rel_(?P<rel>.+)",
        demographics_str,
    )
    if v2_match:
        return frozenset({
            f"country_{_safe_feature_value(v2_match.group('country'))}",
            f"edu_{_safe_feature_value(v2_match.group('edu'))}",
            f"rel_{_safe_feature_value(v2_match.group('rel'))}",
        })

    # Legacy processed_data / processed_data_v1 names.
    # Match longer tokens first: "female" contains "male" as a substring.
    genders = ['female', 'male']
    ages = ['young', 'middle_aged', 'old']
    regions = ['asia', 'europe', 'north_america', 'south_america', 'africa', 'oceania']
    educations = ['edu_low', 'edu_medium', 'edu_high']
    employments = ['employed', 'self_employed', 'unemployed', 'student', 'retired', 'homemaker']
    urban_rural = ['urban', 'rural']

    found_features = []
    for group in [genders, ages, regions, educations, employments, urban_rural]:
        for val in group:
            if val in demographics_str:
                found_features.append(val)
                break
    return frozenset(found_features) if found_features else None


def get_user_features(row: dict, dimensions: Optional[List[str]] = None,
                      *, raw_education: bool = False) -> frozenset:
    """Return the set of demographic feature names for a WVS respondent.

    Args:
        row: WVS data row
        dimensions: List of dimension names to extract. If None, extracts all available dimensions.
                   Valid dimensions: 'gender', 'age', 'age_group', 'region',
                   'country', 'education', 'marital_status', 'religion',
                   'ethnicity', 'employment', 'occupation', 'urban_rural'
    """
    features: Set[str] = set()

    # If no dimensions specified, use default (gender, age, region)
    if dimensions is None:
        dimensions = ['gender', 'age', 'region']

    try:
        # Gender (Q260)
        if 'gender' in dimensions:
            gender = row.get('Q260', '')
            if gender == '1':
                features.add('male')
            elif gender == '2':
                features.add('female')

        # Age (Q262)
        if 'age' in dimensions:
            feat = _age_group_feature(row, legacy=True)
            if feat:
                features.add(feat)

        if 'age_group' in dimensions:
            feat = _age_group_feature(row, legacy=False)
            if feat:
                features.add(feat)

        # Region (B_COUNTRY_ALPHA)
        if 'region' in dimensions:
            country = row.get('B_COUNTRY_ALPHA', '')
            region = COUNTRY_TO_REGION.get(country)
            if region:
                features.add(region)

        if 'country' in dimensions:
            country = _clean_feature_value(row.get('B_COUNTRY_ALPHA'))
            if country:
                features.add(f"country_{_safe_feature_value(country)}")

        # Education (Q275): 0-2 low, 3-5 medium, 6-8 high
        if 'education' in dimensions:
            edu_feature = _education_feature(row, raw=raw_education)
            if edu_feature:
                features.add(edu_feature)

        if 'religion' in dimensions:
            feat = _religion_feature(row)
            if feat:
                features.add(feat)

        if 'marital_status' in dimensions or 'marital' in dimensions:
            feat = _marital_feature(row)
            if feat:
                features.add(feat)

        if 'ethnicity' in dimensions:
            feat = _ethnicity_feature(row)
            if feat:
                features.add(feat)

        # Employment status (Q279): 1=Full time, 2=Part time, 3=Self employed,
        # 4=Retired, 5=Homemaker, 6=Student, 7=Unemployed, 8=Other.
        if 'employment' in dimensions:
            feat = _employment_feature(row)
            if feat:
                features.add(feat)

        if 'occupation' in dimensions:
            feat = _occupation_feature(row)
            if feat:
                features.add(feat)

        # Urban/Rural (H_URBRURAL): 1=Urban, 2=Rural
        if 'urban_rural' in dimensions:
            urb_str = row.get('H_URBRURAL', '')
            if urb_str:
                try:
                    urb = int(float(urb_str))
                    if urb == 1:
                        features.add('urban')
                    elif urb == 2:
                        features.add('rural')
                except (ValueError, TypeError):
                    pass

    except Exception as e:
        print(f"Warning: Error extracting features from row: {e}")

    return frozenset(features)


def get_valid_option_set(question_id: str, nature_options: dict) -> Optional[frozenset]:
    """Return a frozenset of valid integer option keys for a question, or None."""
    try:
        q_info = nature_options.get(question_id)
        if not isinstance(q_info, dict):
            return None
        opts = q_info.get("options", {})
        if not isinstance(opts, dict) or not opts:
            return None
        int_keys = [int(k) for k in opts.keys() if k.lstrip('-').isdigit()]
        if not int_keys:
            return None
        return frozenset(int_keys)
    except Exception as e:
        print(f"Warning: Error getting valid options for question {question_id}: {e}")
        return None

# ======================================================================
# Data Loading
# ======================================================================

def load_wvs_csv(path: str) -> list:
    print(f"Loading WVS CSV from {public_path(path)} ...")
    try:
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        print(f"  {len(rows):,} respondents loaded")
        return rows
    except FileNotFoundError:
        print(f"Error: File not found: {public_path(path)}")
        raise
    except Exception as e:
        print(f"Error loading WVS CSV: {e}")
        raise


def load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {public_path(path)}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {public_path(path)}: {e}")
        raise
    except Exception as e:
        print(f"Error loading JSON file {public_path(path)}: {e}")
        raise


def load_pregenerated_data(training_data_dir: str) -> tuple:
    """Load pre-generated training data from JSON files.

    Args:
        training_data_dir: Directory containing *_train.json files

    Returns:
        tuple: (groups dict, use_augmented bool)
            - groups: dict[frozenset[str], list[dict]] - Grouped QA pairs by demographics
            - use_augmented: bool - Whether data contains augmented reasoning
    """
    groups: Dict[frozenset, list] = defaultdict(list)

    if not os.path.exists(training_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")

    json_files = [f for f in os.listdir(training_data_dir) if f.endswith('_train.json')]

    if not json_files:
        raise ValueError(f"No *_train.json files found in {training_data_dir}")

    print("\nLoading pre-generated training data from "
          f"{public_path(training_data_dir)}")

    # Check first file to determine if data is augmented
    first_file = os.path.join(training_data_dir, json_files[0])
    with open(first_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)

    use_augmented = False
    use_tool_call = False
    if sample_data and isinstance(sample_data[0], dict):
        # Check if it has tool-call format (augmented_data with <tool_call> block)
        if 'augmented_data' in sample_data[0] and '<tool_call>' in sample_data[0].get('augmented_data', ''):
            use_tool_call = True
        # Check if it has 'tool_call_output' field (legacy Qwen native tool-call format)
        elif 'tool_call_output' in sample_data[0]:
            use_tool_call = True
        # Check if it has 'augmented_data' field (augmented format with <answer> tags)
        elif 'augmented_data' in sample_data[0]:
            use_augmented = True

    if use_tool_call:
        fmt_str = "Tool-call (Qwen native format)"
    elif use_augmented:
        fmt_str = "Augmented (with reasoning)"
    else:
        fmt_str = "Clean (direct answers)"
    print(f"Data format detected: {fmt_str}")

    for json_file in tqdm(json_files, desc="Loading training files"):
        features = parse_feature_group_from_filename(json_file)
        if not features:
            print(f"Warning: Could not parse demographics from filename: {json_file}")
            continue

        # Load the JSON data
        file_path = os.path.join(training_data_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_items = json.load(f)

            # Convert to the format expected by the training code
            for item in data_items:
                if use_tool_call:
                    # Tool-call format: augmented_data contains the <tool_call> block
                    tool_call_data = item.get('augmented_data') or item.get('tool_call_output', '')
                    groups[features].append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'augmented_data': tool_call_data,
                    })
                elif use_augmented:
                    # Augmented format: extract answer value from augmented_data
                    answer_match = re.search(r'<answer>(\d+)</answer>', item['augmented_data'])
                    if answer_match:
                        answer_value = answer_match.group(1)
                    else:
                        answer_value = item['answer']

                    # Remove <answer> tags from augmented_data to get the explanation
                    explanation = re.sub(r'<answer>.*?</answer>', '', item['augmented_data'], flags=re.DOTALL).strip()

                    groups[features].append({
                        'question': item['question'],
                        'answer': explanation if explanation else f"Answer: {answer_value}",
                        'answer_value': answer_value,
                    })
                else:
                    # Clean format: answer is just the option number
                    groups[features].append({
                        'question': item['question'],
                        'answer': item['answer'],
                    })

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    print(f"\nLoaded {len(groups)} demographic groups")
    for features, items in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        feature_str = '_'.join(sorted(features))
        print(f"  {feature_str}: {len(items)} samples")

    if len(groups) > 10:
        print(f"  ... and {len(groups) - 10} more groups")

    return dict(groups), use_augmented, use_tool_call


def build_grouped_qa(wvs_data: list, user_indices, question_ids: list,
                     nature_options: dict, *,
                     max_samples_per_group: int = -1,
                     feature_dimensions: Optional[List[str]] = None,
                     raw_education_features: bool = False) -> dict:
    """Build QA pairs grouped by user-feature combination.

    Returns dict[frozenset[str], list[dict]]
    """
    user_set = set(user_indices)
    groups: Dict[frozenset, list] = defaultdict(list)
    skipped_users = 0

    for idx in tqdm(sorted(user_set), desc="Building QA pairs"):
        if idx >= len(wvs_data):
            continue
        row = wvs_data[idx]
        features = get_user_features(
            row, dimensions=feature_dimensions,
            raw_education=raw_education_features)
        if not features:
            skipped_users += 1
            continue

        for qid in question_ids:
            if qid not in nature_options:
                continue
            val = row.get(qid, '')
            if not val:
                continue
            try:
                v = int(float(val))
            except (ValueError, TypeError):
                continue
            if v < 0:
                continue

            q_info = nature_options[qid]
            opts = q_info.get('options', {})
            answer_text = opts.get(str(v))
            if not answer_text:
                continue

            q_text = q_info['question_text']
            opts_fmt = "\n".join(
                f"{k}. {t}" for k, t in
                sorted(opts.items(),
                       key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 0)
            )
            groups[features].append({
                'question_id': qid,
                'question':    f"{q_text}\nOptions:\n{opts_fmt}",
                'answer':       answer_text,
                'answer_value': str(v),
            })

    total = sum(len(v) for v in groups.values())
    print(f"  {total:,} QA pairs in {len(groups)} feature groups "
          f"(skipped {skipped_users} users w/o features)")

    if max_samples_per_group > 0:
        for key in groups:
            if len(groups[key]) > max_samples_per_group:
                groups[key] = random.sample(groups[key],
                                            max_samples_per_group)
        sampled = sum(len(v) for v in groups.values())
        print(f"  After sampling: {sampled:,} QA pairs")

    return dict(groups)

# ======================================================================
# Merge all feature combinations for vLLM usage
# ======================================================================

def merge_all_combinations(wrapper: MultiLoRAModelWrapper,
                           wvs_data: list, user_indices,
                           output_dir: str,
                           feature_dimensions: Optional[List[str]] = None,
                           raw_education_features: bool = False,
                           joint_adapter_groups: bool = False,
                           extra_active_loras: Optional[List[str]] = None,
                           lora_component_mode: str = "full"):
    """Pre-merge LoRA adapters for every unique feature combination
    found among the given users.  Each merged adapter is saved in
    PEFT format and can be loaded by vLLM with a single LoRARequest."""
    try:
        user_set = set(user_indices)
        combos: set = set()
        for idx in user_set:
            if idx < len(wvs_data):
                try:
                    f = get_user_features(
                        wvs_data[idx], dimensions=feature_dimensions,
                        raw_education=raw_education_features)
                    if f:
                        combos.add(f)
                except Exception as e:
                    print(f"Warning: Error getting features for user {idx}: {e}")
                    continue

        print(f"\nMerging {len(combos)} unique feature combinations …")
        os.makedirs(output_dir, exist_ok=True)

        combo_map = {}
        extra_active_loras = list(extra_active_loras or [])
        if lora_component_mode not in {"full", "task_only", "knowledge_only"}:
            raise ValueError(
                "lora_component_mode must be one of: full, task_only, "
                f"knowledge_only; got {lora_component_mode!r}"
            )
        for features in sorted(combos, key=lambda x: sorted(x)):
            try:
                name = "__".join(sorted(features))
                save_dir = os.path.join(output_dir, name)
                if lora_component_mode == "task_only":
                    active_names = list(extra_active_loras)
                else:
                    active_names = (
                        [joint_adapter_name(features)]
                        if joint_adapter_groups else list(features)
                    )
                    if lora_component_mode == "full":
                        active_names = list(extra_active_loras) + active_names
                wrapper.merge_and_save(active_names, save_dir)
                combo_map[name] = list(sorted(features))
            except Exception as e:
                print(f"Warning: Error merging features {features}: {e}")
                continue

        try:
            with open(os.path.join(output_dir, "combo_map.json"), 'w') as f:
                json.dump(combo_map, f, indent=2, ensure_ascii=False)
            print("Combo map saved -> "
                  f"{public_path(os.path.join(output_dir, 'combo_map.json'))}")
        except Exception as e:
            print(f"Warning: Error saving combo map: {e}")

    except Exception as e:
        print(f"Error during merge_all_combinations: {e}")
        raise
