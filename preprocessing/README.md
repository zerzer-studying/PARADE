# Data preprocessing

The training entry points materialize QA examples directly from the released
WVS CSV and SocioBench JSON files. WVS feature extraction and QA construction
are implemented in `src/utils.py`. The corresponding SocioBench integration is
implemented in `sociobench_code/utils.py` and
`sociobench_code/train_eval.py`.

This directory contains the scripts that rebuild the held-out respondent
manifests used for validation and the five main test repetitions. They enforce
the same eight complete demographic dimensions used by the experiments.

## Locked base artifacts

Two historical, fixed partitions are committed as protocol inputs:

- `data_splits/wvs/questions_split.json` and
  `data_splits/wvs/train_users.json` define the WVS question partition and the
  initial 96,220/1,000 respondent pool split.
- `data_splits/sociobench/main_base.json` defines the matched SocioBench
  training, held-out respondent, and question partitions shared by all runs.

The generation scripts do not resample these base artifacts. They regenerate
the disjoint validation users and complete-profile test users from them.

## Rebuild and verify

Run from the repository root after placing the raw datasets as described in
the main README:

```bash
python preprocessing/build_wvs_splits.py \
  --wvs-csv data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv

python preprocessing/build_sociobench_splits.py \
  --sociobench-root data/SocioBench
```

The default output is `generated_data_splits/`, so the committed protocol is
never overwritten. Verify both the invariants and exact agreement with the
committed manifests:

```bash
python preprocessing/verify_splits.py \
  --candidate-root generated_data_splits \
  --reference-root data_splits
```

The WVS validation split contains 100 respondents. Each WVS test repetition
contains 100 respondents. SocioBench validation contains 50 respondents per
domain, and each test repetition contains 100 respondents per domain. Test
seeds are 42, 43, 44, 45, and 46; the validation selection seed is 31415.
