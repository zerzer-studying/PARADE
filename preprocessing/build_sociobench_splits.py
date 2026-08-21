#!/usr/bin/env python3
"""Rebuild SocioBench validation and five-seed test manifests."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sociobench_code.utils import (  # noqa: E402
    DEFAULT_FEATURE_DIMENSIONS,
    get_sociobench_user_features,
    load_sociobench_ground_truth,
)


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
            "Build complete-eight-feature SocioBench validation and test "
            "manifests from the committed matched base split."
        )
    )
    parser.add_argument(
        "--sociobench-root",
        type=Path,
        default=PROJECT_ROOT / "data/SocioBench",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=PROJECT_ROOT / "data_splits/sociobench/main_base.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "generated_data_splits/sociobench",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=list(FINAL_TEST_SEEDS),
    )
    parser.add_argument("--test-users-per-domain", type=int, default=100)
    parser.add_argument("--validation-seed", type=int, default=31415)
    parser.add_argument("--validation-users-per-domain", type=int, default=50)
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
    if feature.startswith("age_"):
        return "age_group"
    if feature.startswith("country_"):
        return "country"
    if feature.startswith("rel_"):
        return "religion"
    if feature.startswith("edu_"):
        return "education"
    if feature.startswith("marital_"):
        return "marital_status"
    if feature.startswith("employment_"):
        return "employment"
    if feature in {"urban", "rural"}:
        return "urban_rural"
    return None


def has_complete_profile(features: frozenset[str]) -> bool:
    present = {
        dimension
        for feature in features
        if (dimension := feature_dimension(feature)) is not None
    }
    return set(FEATURE_DIMENSIONS) <= present


def main() -> None:
    args = parse_args()
    base = load_json(args.base_manifest)
    if list(base.get("feature_dimensions", [])) != list(DEFAULT_FEATURE_DIMENSIONS):
        raise ValueError(
            "The base manifest feature dimensions differ from the evaluator"
        )

    complete_held_out: dict[str, list[str]] = {}
    validation_ids: dict[str, list[str]] = {}
    for domain, split in base["domains"].items():
        domain_id = int(split["domain_id"])
        respondents = load_sociobench_ground_truth(
            str(args.sociobench_root), domain, int(base["dataset_size"])
        )
        by_id = {str(row["person_id"]): row for row in respondents}
        complete_held_out[domain] = [
            str(person_id)
            for person_id in split["held_out_person_ids"]
            if (
                str(person_id) in by_id
                and has_complete_profile(
                    get_sociobench_user_features(
                        by_id[str(person_id)].get("attributes", {}),
                        domain_id,
                        list(FEATURE_DIMENSIONS),
                    )
                )
            )
        ]
        count = args.validation_users_per_domain
        if len(complete_held_out[domain]) < count:
            raise ValueError(
                f"{domain}: not enough complete held-out respondents"
            )
        validation_ids[domain] = random.Random(
            args.validation_seed + domain_id
        ).sample(sorted(complete_held_out[domain]), count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "main_base.json", base)

    final_eval_ids = {domain: set() for domain in base["domains"]}
    test_summary = {}
    for seed in args.test_seeds:
        manifest = copy.deepcopy(base)
        manifest["eval_sampling_seed"] = seed
        manifest["protocol_role"] = "main_test"
        manifest["requires_complete_feature_dimensions"] = 8
        manifest["incomplete_user_replacements"] = {}
        seed_summary = {}

        for domain, split in manifest["domains"].items():
            held_out = [str(value) for value in split["held_out_person_ids"]]
            original = random.Random(seed).sample(
                held_out, args.test_users_per_domain
            )
            complete_set = set(complete_held_out[domain])
            validation_set = set(validation_ids[domain])
            selected = [
                person_id
                for person_id in original
                if person_id in complete_set and person_id not in validation_set
            ]
            removed = [
                person_id
                for person_id in original
                if person_id not in complete_set or person_id in validation_set
            ]
            replacements = random.Random(
                seed + 1000 * int(split["domain_id"])
            ).sample(
                sorted(complete_set - validation_set - set(selected)),
                len(removed),
            )
            selected.extend(replacements)
            if len(selected) != len(set(selected)):
                raise RuntimeError(
                    f"{domain}: duplicate respondents for test seed {seed}"
                )

            split["eval_person_ids"] = selected
            manifest["incomplete_user_replacements"][domain] = {
                "count": len(removed),
                "removed_person_ids": removed,
                "replacement_person_ids": replacements,
            }
            final_eval_ids[domain].update(selected)
            seed_summary[domain] = {
                "test_users": len(selected),
                "replacements": len(replacements),
            }

        dump_json(args.output_dir / f"test_users_seed{seed}.json", manifest)
        test_summary[str(seed)] = seed_summary

    validation = copy.deepcopy(base)
    validation["protocol_role"] = "lambda_validation"
    validation["selection_seed"] = args.validation_seed
    validation["final_test_seeds_excluded"] = list(args.test_seeds)
    validation["requires_complete_feature_dimensions"] = 8
    for domain, split in validation["domains"].items():
        split["eval_person_ids"] = validation_ids[domain]
        overlap = set(split["eval_person_ids"]) & final_eval_ids[domain]
        if overlap:
            raise RuntimeError(
                f"{domain}: validation overlaps final test respondents"
            )
    dump_json(args.output_dir / "lambda_validation.json", validation)

    summary = {
        "domains": len(base["domains"]),
        "validation_seed": args.validation_seed,
        "validation_users_per_domain": args.validation_users_per_domain,
        "test_seeds": list(args.test_seeds),
        "test_users_per_domain": args.test_users_per_domain,
        "feature_dimensions": list(FEATURE_DIMENSIONS),
        "test_splits": test_summary,
    }
    dump_json(args.output_dir / "split_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
