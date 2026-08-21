#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export MODEL_KEY=ministral
export MODEL_PATH=${MODEL_PATH:-"models/Ministral-3-8B-Instruct-2512-BF16"}
export SOFT_LAMBDA=0.5
export INFERENCE_TASK_SCALE=0.875
export INFERENCE_KNOWLEDGE_SCALE=1.0
export LORA_RANK=4
export LORA_ALPHA=8
export TARGET_MODULES_STR="gate_proj up_proj down_proj q_proj k_proj v_proj o_proj"
export TASK_ANSWER_MODE=balanced_random
export TASK_BALANCE_ANSWERS=1
export TASK_MICRO_BATCH_SIZE=2
export KNOWLEDGE_MICRO_BATCH_SIZE=2
export KNOWLEDGE_GRAD_ACCUM=2
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-29630}

exec bash "$PROJECT_ROOT/scripts/lib/run_sociobench.sh" "$@"
