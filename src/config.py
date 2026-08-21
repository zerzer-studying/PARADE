import os
import json
import csv
import gc
import math
import time
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, OrderedDict
import numpy as np
import random

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

# ======================================================================
# Feature Configuration
# ======================================================================

FEATURE_DIMENSIONS = OrderedDict([
    ('gender',    ['male', 'female']),
    ('age',       ['young', 'middle_aged', 'old']),
    ('age_group', ['age_young', 'age_middle', 'age_senior']),
    ('region',    ['asia', 'europe', 'north_america',
                   'south_america', 'africa', 'oceania']),
    ('education', ['edu_low', 'edu_medium', 'edu_high']),
    ('marital_status', ['marital_married', 'marital_living_together',
                        'marital_divorced', 'marital_separated',
                        'marital_widowed', 'marital_single']),
    ('religion', ['rel_0', 'rel_1', 'rel_2', 'rel_3', 'rel_4',
                  'rel_5', 'rel_6', 'rel_7', 'rel_8']),
    ('employment', ['employment_full_time', 'employment_part_time',
                    'employment_self_employed', 'employment_retired',
                    'employment_homemaker', 'employment_student',
                    'employment_unemployed', 'employment_other']),
    ('occupation', [f'occupation_{i}' for i in range(12)]),
    ('urban_rural', ['urban', 'rural']),
])

ALL_FEATURES: List[str] = []
for _feats in FEATURE_DIMENSIONS.values():
    ALL_FEATURES.extend(_feats)

TARGET_MODULES = ['gate_proj', 'up_proj', 'down_proj']

REGION_COUNTRIES = {
    'asia': [
        'CHN', 'JPN', 'KOR', 'TWN', 'HKG', 'MAC', 'MNG',
        'IND', 'PAK', 'BGD', 'LKA', 'NPL', 'BTN', 'MDV',
        'IDN', 'THA', 'VNM', 'PHL', 'MMR', 'MYS', 'SGP',
        'KHM', 'LAO', 'BRN', 'TLS',
        'KAZ', 'UZB', 'TJK', 'KGZ', 'TKM',
        'TUR', 'IRN', 'IRQ', 'SAU', 'YEM', 'SYR', 'JOR',
        'LBN', 'ISR', 'PSE', 'ARE', 'OMN', 'KWT', 'QAT', 'BHR',
    ],
    'europe': [
        'DEU', 'GBR', 'FRA', 'ESP', 'ITA', 'NLD', 'SWE', 'NOR',
        'DNK', 'FIN', 'POL', 'ROU', 'GRC', 'PRT', 'CZE', 'HUN',
        'BGR', 'HRV', 'SVK', 'SVN', 'EST', 'LVA', 'LTU', 'CYP',
        'AUT', 'BEL', 'CHE', 'ISL', 'IRL', 'LUX', 'MLT', 'MKD',
        'SRB', 'BIH', 'MNE', 'ALB', 'UKR', 'BLR', 'RUS', 'ARM',
        'GEO', 'AZE', 'AND', 'NIR',
    ],
    'north_america': [
        'USA', 'CAN', 'MEX', 'GTM', 'BLZ', 'SLV', 'HND',
        'NIC', 'CRI', 'PAN', 'CUB', 'JAM', 'HTI', 'DOM',
        'PRI', 'TTO', 'BHS', 'BRB',
    ],
    'south_america': [
        'BRA', 'ARG', 'CHL', 'COL', 'PER', 'VEN', 'ECU',
        'BOL', 'PRY', 'URY', 'GUY', 'SUR', 'GUF',
    ],
    'africa': [
        'EGY', 'LBY', 'TUN', 'DZA', 'MAR', 'SDN', 'SSD',
        'NGA', 'GHA', 'CIV', 'SEN', 'MLI', 'BFA', 'NER',
        'GNB', 'GIN', 'SLE', 'LBR', 'TGO', 'BEN', 'GMB', 'MRT',
        'ETH', 'KEN', 'TZA', 'UGA', 'RWA', 'BDI', 'SOM',
        'ERI', 'DJI', 'COD', 'AGO', 'CMR', 'TCD', 'CAF',
        'COG', 'GAB', 'GNQ', 'STP', 'ZAF', 'ZWE', 'ZMB',
        'MWI', 'MOZ', 'BWA', 'NAM', 'LSO', 'SWZ',
    ],
    'oceania': [
        'AUS', 'NZL', 'PNG', 'FJI', 'SLB', 'VUT', 'NCL',
        'PYF', 'WSM', 'KIR', 'TON', 'PLW', 'FSM', 'MHL',
        'NRU', 'TUV',
    ],
}

COUNTRY_TO_REGION: Dict[str, str] = {}
for _region, _countries in REGION_COUNTRIES.items():
    for _c in _countries:
        COUNTRY_TO_REGION[_c] = _region
