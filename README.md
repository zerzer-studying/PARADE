# PARADE

PARADE uses parametric demographic injection to encode demographics into
reusable and composable parameters, enabling generalization to unseen profiles
and better alignment with real survey distributions.

This repository contains the implementation and reproducibility scripts for
the PARADE main experiments on WVS and SocioBench. PARADE first trains a shared
task LoRA and then trains eight demographic LoRA families with residual-gate
weights and factor-level soft orthogonality regularization.

The repository includes our method, locked data splits, preprocessing tools,
and the final configurations for three backbones. Raw datasets, model weights,
trained checkpoints, baseline implementations, and result files are not
included.

## Requirements

The code was tested with:

- Python 3.13
- PyTorch 2.10
- CUDA 12.8
- Transformers 5.10
- four 80 GB NVIDIA GPUs

Create an isolated environment from the repository root:

```bash
cd PARADE
conda create --name parade python=3.13 -y
conda activate parade
pip install --upgrade pip
pip install -r requirements.txt
```

`flash-linear-attention`, `fla-core`, and `causal-conv1d` provide the optional
Qwen fast path and require a compatible CUDA compiler. WVS Llama and Qwen
evaluation uses vLLM by default. Set `USE_VLLM_EVAL=0` to use the Hugging Face
evaluator instead.

### Reference hardware

| Component | Configuration |
|---|---|
| GPU | 4 x NVIDIA A800-SXM4-80GB (81,920 MiB each) |
| CPU | 2 x Intel Xeon Platinum 8358 at 2.60 GHz |
| Operating system | Ubuntu 20.04.6 LTS, Linux 4.18.0 |

## Main Components

- `src/`: shared LoRA implementation, WVS data loading, training, and
  evaluation.
- `sociobench_code/`: SocioBench feature mapping, data construction, training,
  and evaluation.
- `scripts/wvs/`: final WVS configurations for Llama, Qwen, and Ministral.
- `scripts/sociobench/`: final SocioBench configurations for the three
  backbones.
- `scripts/lib/`: shared two-stage training and evaluation drivers.
- `preprocessing/`: scripts for rebuilding and validating respondent splits.
- `data_splits/`: locked question, training, validation, and five-seed test
  partitions.

```text
PARADE/
├── src/
├── sociobench_code/
├── preprocessing/
├── scripts/
│   ├── lib/
│   ├── wvs/
│   └── sociobench/
├── data_splits/
│   ├── wvs/
│   └── sociobench/
├── data/wvs/nature_options.json
├── requirements.txt
└── README.md
```

## Running Experiments

Run all commands below from the `PARADE/` repository root.

### 0. Prepare datasets and models

Download the WVS Wave 7 v6.0 CSV and the official SocioBench dataset, then
place the datasets and model weights in the following layout:

```text
data/
├── wvs/
│   ├── WVS_Cross-National_Wave_7_csv_v6_0.csv
│   └── nature_options.json
└── SocioBench/
    └── Dataset_all/
        ├── q&a/
        └── A_GroundTruth_sampling500/

models/
├── Meta-Llama-3.1-8B-Instruct/
├── Qwen3.5-9B/
└── Ministral-3-8B-Instruct-2512-BF16/
```

The experiments use the same eight complete demographic dimensions on both
datasets: gender, age group, country, religion, education, marital status,
employment, and urban/rural residence.

### 1. Preprocessing

The committed `data_splits/` files can be used directly to reproduce the main
experiments. Training QA examples are constructed from the released datasets
at runtime, so no separate materialized training dataset is required.

To validate the committed splits:

```bash
python preprocessing/verify_splits.py
```

To rebuild the validation and seeds 42-46 test-user splits:

```bash
python preprocessing/build_wvs_splits.py \
  --wvs-csv data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv

python preprocessing/build_sociobench_splits.py \
  --sociobench-root data/SocioBench

python preprocessing/verify_splits.py \
  --candidate-root generated_data_splits \
  --reference-root data_splits
```

Generated files are written to `generated_data_splits/`; committed splits are
not overwritten. See `preprocessing/README.md` for the locked base artifacts,
selection seeds, split sizes, and verification rules.

### 2. Training

Each model script accepts an action and a comma-separated GPU list:

```text
bash scripts/<dataset>/<backbone>.sh <all|train|eval> <GPU_IDS>
```

Train both stages without running final evaluation:

```bash
bash scripts/wvs/llama.sh train 0,1,2,3
bash scripts/sociobench/llama.sh train 0,1,2,3
```

Stage I trains the shared task LoRA. Stage II freezes the task LoRA and trains
the demographic LoRAs, residual-gate weights, and soft orthogonality
regularization objective. Training saves resumable checkpoints and
automatically resumes an incomplete stage when the command is rerun.

### 3. Evaluation

Evaluate an existing trained configuration on test-user seeds 42-46:

```bash
bash scripts/wvs/llama.sh eval 0,1,2,3
bash scripts/sociobench/llama.sh eval 0,1,2,3
```

Run training and evaluation end to end for all six final configurations:

```bash
# WVS
bash scripts/wvs/llama.sh all 0,1,2,3
bash scripts/wvs/qwen.sh all 0,1,2,3
bash scripts/wvs/ministral.sh all 0,1,2,3

# SocioBench
bash scripts/sociobench/llama.sh all 0,1,2,3
bash scripts/sociobench/qwen.sh all 0,1,2,3
bash scripts/sociobench/ministral.sh all 0,1,2,3
```

Override model or dataset locations without editing a script:

```bash
MODEL_PATH=models/custom-model WVS_CSV=data/wvs/custom.csv \
  bash scripts/wvs/llama.sh all 0,1,2,3

MODEL_PATH=models/custom-model SOCIOBENCH_ROOT=data/SocioBench \
  bash scripts/sociobench/llama.sh all 0,1,2,3
```

## Final Configurations

All configurations use LoRA rank 4, alpha 8, dropout 0.05, one epoch per
stage, and the following MLP and attention targets:

```text
gate_proj up_proj down_proj q_proj k_proj v_proj o_proj
```

| Dataset | Backbone | Lambda | Inference task scale | Inference knowledge scale |
|---|---:|---:|---:|---:|
| WVS | Llama | 0.3 | 0.5 | 1.0 |
| WVS | Qwen | 0.5 | 1.0 | 0.75 |
| WVS | Ministral | 0.3 | 1.0 | 1.1 |
| SocioBench | Llama | 0.4 | 0.75 | 0.66 |
| SocioBench | Qwen | 0.7 | 0.75 | 1.15 |
| SocioBench | Ministral | 0.5 | 0.875 | 1.0 |

Lambda and the inference-time task and knowledge scales were selected on the
held-out respondent validation split. The five final test-user seeds were
evaluated only after these hyperparameters were fixed.

## Outputs

Results and checkpoints are written to:

```text
outputs/<dataset>/<backbone>/
├── task/
├── knowledge_lambda_<value>/
└── main/
    ├── seed_42/
    ├── seed_43/
    ├── seed_44/
    ├── seed_45/
    └── seed_46/
```

Each seed directory contains:

- `eval_results.json`: Accuracy, MAE, JS divergence, KL divergence, and EMD.
- `run_timing.json`: runtime and hardware metadata.
- `run.log`: the complete evaluation log.

Saved hardware metadata contains only GPU properties. Repository-local paths
are recorded as relative paths, and external absolute paths are replaced with
`<external-path>` in public configuration artifacts and argument summaries.

Metrics are computed from respondent-level predictions and the corresponding
response distributions aggregated within each question and demographic group.
