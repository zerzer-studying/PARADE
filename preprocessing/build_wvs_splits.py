#!/usr/bin/env python3
"""Rebuild the WVS validation and five-seed test-user splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import get_user_features  # noqa: E402


FEATURE_DIMENSIONS = (
    "gender",
    "age_group",
    "country",
    "religion",
    "education",
    "marital_status",
    "employment",
    "urban_rural",
)
FINAL_TEST_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build complete-eight-feature WVS validation and test-user splits "
            "from the committed base user pool."
        )
    )
    parser.add_argument(
        "--wvs-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv",
    )
    parser.add_argument(
        "--base-user-split",
        type=Path,
        default=PROJECT_ROOT / "data_splits/wvs/train_users.json",
    )
    parser.add_argument(
        "--question-split",
        type=Path,
        default=PROJECT_ROOT / "data_splits/wvs/questions_split.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "generated_data_splits/wvs",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=list(FINAL_TEST_SEEDS),
    )
    parser.add_argument("--test-users", type=int, default=100)
    parser.add_argument("--validation-seed", type=int, default=31415)
    parser.add_argument("--validation-users", type=int, default=100)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def feature_dimension(feature: str) -> str | None:
    if feature in {"male", "female"}:
        return "gender"
    prefixes = {
        "age_": "age_group",
        "country_": "country",
        "rel_": "religion",
        "edu_": "education",
        "marital_": "marital_status",
        "employment_": "employment",
    }
    for prefix, dimension in prefixes.items():
        if feature.startswith(prefix):
            return dimension
    if feature in {"urban", "rural"}:
        return "urban_rural"
    return None


def main() -> None:
    args = parse_args()
    base = load_json(args.base_user_split)
    questions = load_json(args.question_split)
    with args.wvs_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(base["train"]) + len(base["test"]):
        raise ValueError(
            "The WVS CSV row count does not match the committed base user split"
        )

    complete_cache: dict[int, bool] = {}

    def is_complete(user_id: int) -> bool:
        if user_id < 0 or user_id >= len(rows):
            return False
        if user_id not in complete_cache:
            features = get_user_features(
                rows[user_id],
                dimensions=list(FEATURE_DIMENSIONS),
                raw_education=True,
            )
            present = {
                dimension
                for feature in features
                if (dimension := feature_dimension(feature)) is not None
            }
            complete_cache[user_id] = set(FEATURE_DIMENSIONS) <= present
        return complete_cache[user_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "train_users.json", base)
    dump_json(args.output_dir / "questions_split.json", questions)

    final_test_ids: set[int] = set()
    test_summaries = {}
    for seed in args.test_seeds:
        original = random.Random(seed).sample(base["test"], args.test_users)
        incomplete = [user_id for user_id in original if not is_complete(user_id)]
        original_set = set(original)
        eligible = [
            user_id
            for user_id in base["test"]
            if user_id not in original_set and is_complete(user_id)
        ]
        replacements = random.Random(seed).sample(eligible, len(incomplete))
        replacement_by_user = dict(zip(incomplete, replacements))
        selected = [
            replacement_by_user.get(user_id, user_id) for user_id in original
        ]
        if len(selected) != len(set(selected)) or not all(map(is_complete, selected)):
            raise RuntimeError(f"Generated invalid WVS test split for seed {seed}")

        payload = {
            "metadata": {
                "source": "data_splits/wvs/train_users.json",
                "role": "test_selected_seed_screen",
                "selection_seed": seed,
                "selection_protocol": (
                    "random sample then deterministic replacement of "
                    "incomplete users"
                ),
                "complete_feature_dimensions": list(FEATURE_DIMENSIONS),
                "replaced_user_ids": incomplete,
                "replacement_user_ids": replacements,
                "test_count": len(selected),
                "protocol_role": "main_test",
                "question_split": "data_splits/wvs/questions_split.json",
            },
            "train": base["train"],
            "test": selected,
        }
        dump_json(args.output_dir / f"test_users_seed{seed}.json", payload)
        final_test_ids.update(selected)
        test_summaries[str(seed)] = {
            "test_users": len(selected),
            "replacements": len(replacements),
        }

    complete_pool = [user_id for user_id in base["test"] if is_complete(user_id)]
    validation_candidates = sorted(set(complete_pool) - final_test_ids)
    if len(validation_candidates) < args.validation_users:
        raise ValueError("Not enough complete users for the validation split")
    validation_users = random.Random(args.validation_seed).sample(
        validation_candidates, args.validation_users
    )
    validation = {
        "metadata": {
            "role": "lambda_validation",
            "source": "data_splits/wvs/train_users.json",
            "selection_seed": args.validation_seed,
            "excluded_final_test_seeds": list(args.test_seeds),
            "complete_feature_dimensions": list(FEATURE_DIMENSIONS),
            "test_count": len(validation_users),
            "question_split": "data_splits/wvs/questions_split.json",
        },
        "train": base["train"],
        "test": validation_users,
    }
    dump_json(args.output_dir / "validation_users.json", validation)

    summary = {
        "wvs_csv_rows": len(rows),
        "base_train_users": len(base["train"]),
        "base_held_out_users": len(base["test"]),
        "question_counts": {
            "train": len(questions["train"]),
            "test": len(questions["test"]),
        },
        "validation_seed": args.validation_seed,
        "validation_users": len(validation_users),
        "validation_test_overlap": len(set(validation_users) & final_test_ids),
        "test_splits": test_summaries,
        "feature_dimensions": list(FEATURE_DIMENSIONS),
    }
    dump_json(args.output_dir / "split_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
