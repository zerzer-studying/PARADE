#!/usr/bin/env python3
"""Validate PARADE data splits and optionally compare regenerated files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=PROJECT_ROOT / "data_splits",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="When set, require candidate JSON payloads to match this root.",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_wvs(root: Path) -> dict:
    wvs = root / "wvs"
    base = load(wvs / "train_users.json")
    questions = load(wvs / "questions_split.json")
    validation = load(wvs / "validation_users.json")

    assert len(questions["train"]) == 207
    assert len(questions["test"]) == 51
    train_question_ids = {item["id"] for item in questions["train"]}
    test_question_ids = {item["id"] for item in questions["test"]}
    assert train_question_ids.isdisjoint(test_question_ids)

    assert len(base["train"]) == len(set(base["train"])) == 96220
    assert len(base["test"]) == len(set(base["test"])) == 1000
    assert set(base["train"]).isdisjoint(base["test"])

    all_test_users: set[int] = set()
    for seed in FINAL_TEST_SEEDS:
        split = load(wvs / f"test_users_seed{seed}.json")
        assert split["train"] == base["train"]
        assert len(split["test"]) == len(set(split["test"])) == 100
        assert set(split["test"]) <= set(base["test"])
        assert (
            split["metadata"]["complete_feature_dimensions"]
            == validation["metadata"]["complete_feature_dimensions"]
        )
        all_test_users.update(split["test"])

    assert validation["train"] == base["train"]
    assert len(validation["test"]) == len(set(validation["test"])) == 100
    assert set(validation["test"]) <= set(base["test"])
    assert set(validation["test"]).isdisjoint(all_test_users)
    return {
        "train_questions": len(questions["train"]),
        "test_questions": len(questions["test"]),
        "base_train_users": len(base["train"]),
        "base_held_out_users": len(base["test"]),
        "validation_users": len(validation["test"]),
        "test_seeds": list(FINAL_TEST_SEEDS),
    }


def verify_sociobench(root: Path) -> dict:
    socio = root / "sociobench"
    base = load(socio / "main_base.json")
    validation = load(socio / "lambda_validation.json")
    all_eval_ids = {domain: set() for domain in base["domains"]}

    for seed in FINAL_TEST_SEEDS:
        manifest = load(socio / f"test_users_seed{seed}.json")
        assert manifest["eval_sampling_seed"] == seed
        assert manifest["protocol_role"] == "main_test"
        assert manifest["feature_dimensions"] == base["feature_dimensions"]
        for domain, base_split in base["domains"].items():
            split = manifest["domains"][domain]
            assert split["train_person_ids"] == base_split["train_person_ids"]
            assert split["held_out_person_ids"] == base_split["held_out_person_ids"]
            assert split["train_question_ids"] == base_split["train_question_ids"]
            assert split["test_question_ids"] == base_split["test_question_ids"]
            eval_ids = list(map(str, split["eval_person_ids"]))
            assert len(eval_ids) == len(set(eval_ids)) == 100
            assert set(eval_ids).isdisjoint(
                map(str, split["train_person_ids"])
            )
            all_eval_ids[domain].update(eval_ids)

    assert validation["protocol_role"] == "lambda_validation"
    for domain, base_split in base["domains"].items():
        split = validation["domains"][domain]
        assert split["train_person_ids"] == base_split["train_person_ids"]
        assert split["held_out_person_ids"] == base_split["held_out_person_ids"]
        assert split["train_question_ids"] == base_split["train_question_ids"]
        assert split["test_question_ids"] == base_split["test_question_ids"]
        validation_ids = list(map(str, split["eval_person_ids"]))
        assert len(validation_ids) == len(set(validation_ids)) == 50
        assert set(validation_ids).isdisjoint(all_eval_ids[domain])

    return {
        "domains": len(base["domains"]),
        "train_users_per_domain": len(
            next(iter(base["domains"].values()))["train_person_ids"]
        ),
        "validation_users_per_domain": 50,
        "test_users_per_domain": 100,
        "test_seeds": list(FINAL_TEST_SEEDS),
    }


def compare_payloads(candidate: Path, reference: Path) -> list[str]:
    mismatches = []
    for dataset in ("wvs", "sociobench"):
        names = [
            "validation_users.json"
            if dataset == "wvs"
            else "lambda_validation.json",
            *[f"test_users_seed{seed}.json" for seed in FINAL_TEST_SEEDS],
        ]
        for name in names:
            candidate_path = candidate / dataset / name
            reference_path = reference / dataset / name
            if load(candidate_path) != load(reference_path):
                mismatches.append(str(Path(dataset) / name))
    return mismatches


def main() -> None:
    args = parse_args()
    summary = {
        "wvs": verify_wvs(args.candidate_root),
        "sociobench": verify_sociobench(args.candidate_root),
    }
    if args.reference_root is not None:
        mismatches = compare_payloads(args.candidate_root, args.reference_root)
        if mismatches:
            raise ValueError(
                "Regenerated split payloads differ from the reference: "
                + ", ".join(mismatches)
            )
        summary["reference_comparison"] = "exact JSON payload match"
    summary["status"] = "passed"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
