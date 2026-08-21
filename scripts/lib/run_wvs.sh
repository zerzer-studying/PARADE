#!/usr/bin/env bash

set -euo pipefail

ACTION=${1:-all}
GPU_IDS=${2:-0,1,2,3}

case "$ACTION" in
  all|train|eval) ;;
  *)
    echo "Usage: bash <model-script> [all|train|eval] [GPU_IDS]" >&2
    exit 2
    ;;
esac

: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${SOFT_LAMBDA:?SOFT_LAMBDA is required}"
: "${INFERENCE_TASK_SCALE:?INFERENCE_TASK_SCALE is required}"
: "${INFERENCE_KNOWLEDGE_SCALE:?INFERENCE_KNOWLEDGE_SCALE is required}"
: "${TASK_MICRO_BATCH_SIZE:?TASK_MICRO_BATCH_SIZE is required}"
: "${KNOWLEDGE_MICRO_BATCH_SIZE:?KNOWLEDGE_MICRO_BATCH_SIZE is required}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/wvs/$MODEL_KEY"}
WVS_CSV=${WVS_CSV:-"data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv"}
NATURE_OPTIONS=${NATURE_OPTIONS:-"data/wvs/nature_options.json"}
QUESTIONS_SPLIT=${QUESTIONS_SPLIT:-"data_splits/wvs/questions_split.json"}
TRAIN_USERS_SPLIT=${TRAIN_USERS_SPLIT:-"data_splits/wvs/train_users.json"}
VALIDATION_USERS_SPLIT=${VALIDATION_USERS_SPLIT:-"data_splits/wvs/validation_users.json"}

LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
TARGET_MODULES_STR=${TARGET_MODULES_STR:-"gate_proj up_proj down_proj q_proj k_proj v_proj o_proj"}
FEATURE_DIMENSIONS_STR=${FEATURE_DIMENSIONS_STR:-"gender age_group country religion education marital_status employment urban_rural"}
TASK_MAX_SAMPLES=${TASK_MAX_SAMPLES:-12000}
MAX_TRAIN_USERS=${MAX_TRAIN_USERS:-700}
MAX_SAMPLES_PER_GROUP=${MAX_SAMPLES_PER_GROUP:--1}
TASK_EPOCHS=${TASK_EPOCHS:-1}
KNOWLEDGE_EPOCHS=${KNOWLEDGE_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_WEIGHT_LEARNING_RATE=${LORA_WEIGHT_LEARNING_RATE:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
MAX_EVAL_USERS=${MAX_EVAL_USERS:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29510}
USE_VLLM_EVAL=${USE_VLLM_EVAL:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}

read -r -a TARGET_MODULES <<< "$TARGET_MODULES_STR"
read -r -a FEATURE_DIMENSIONS <<< "$FEATURE_DIMENSIONS_STR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
LAMBDA_TAG=${SOFT_LAMBDA//./p}
TASK_DIR="$OUTPUT_ROOT/task"
KNOWLEDGE_DIR="$OUTPUT_ROOT/knowledge_lambda_${LAMBDA_TAG}"

for required_path in \
  "$MODEL_PATH" "$WVS_CSV" "$NATURE_OPTIONS" "$QUESTIONS_SPLIT" \
  "$TRAIN_USERS_SPLIT" "$VALIDATION_USERS_SPLIT"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --wvs_csv "$WVS_CSV"
  --nature_options "$NATURE_OPTIONS"
  --questions_split "$QUESTIONS_SPLIT"
  --feature_dimensions "${FEATURE_DIMENSIONS[@]}"
  --raw_education_features
  --require_complete_feature_dimensions
  --model_name "$MODEL_PATH"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "${TARGET_MODULES[@]}"
  --lora_dimension_order "${FEATURE_DIMENSIONS[@]}"
  --task_lora_name task_shared
  --seed 42
  --batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --max_length 1024
  --max_train_users "$MAX_TRAIN_USERS"
  --train_answer_value_only
)

run_training() {
  mkdir -p "$TASK_DIR" "$KNOWLEDGE_DIR"

  if [[ -d "$TASK_DIR/final/task_shared" ]]; then
    echo "Stage I already complete: $TASK_DIR/final/task_shared"
  else
    echo "Stage I: training the demographic-agnostic task LoRA"
    "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT_BASE" \
      src/train_eval.py \
      "${COMMON_ARGS[@]}" \
      --users_split "$TRAIN_USERS_SPLIT" \
      --output_dir "$TASK_DIR" \
      --lora_weight_mode uniform \
      --lora_composition_mode sum \
      --lora_orthogonal_strength 0 \
      --lora_soft_orthogonal_lambda 0 \
      --lora_task_scale 1 \
      --lora_knowledge_scale 1 \
      --lora_train_stage task_only \
      --task_train_data_mode global_random \
      --task_answer_mode balanced_random \
      --task_balance_answers \
      --task_max_samples "$TASK_MAX_SAMPLES" \
      --max_samples_per_group -1 \
      --num_epochs "$TASK_EPOCHS" \
      --train_micro_batch_size "$TASK_MICRO_BATCH_SIZE" \
      --dynamic_train_padding \
      --learning_rate "$LEARNING_RATE" \
      --num_checkpoints "$NUM_CHECKPOINTS" \
      --mode train_only \
      2>&1 | tee "$TASK_DIR/run.log"
  fi

  if [[ -d "$KNOWLEDGE_DIR/final" ]]; then
    echo "Stage II already complete: $KNOWLEDGE_DIR/final"
  else
    echo "Stage II: training demographic LoRAs with soft orthogonality"
    "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$((MASTER_PORT_BASE + 1))" \
      src/train_eval.py \
      "${COMMON_ARGS[@]}" \
      --users_split "$VALIDATION_USERS_SPLIT" \
      --output_dir "$KNOWLEDGE_DIR" \
      --lora_weight_mode residual_gate \
      --lora_weight_normalize sum_to_active_count \
      --lora_composition_mode explicit_task_knowledge_projection \
      --lora_orthogonal_strength 0 \
      --lora_soft_orthogonal_lambda "$SOFT_LAMBDA" \
      --lora_task_scale 1 \
      --lora_knowledge_scale 1 \
      --lora_train_stage knowledge_only \
      --load_task_lora_dir "$TASK_DIR/final" \
      --freeze_task_lora \
      --max_samples_per_group "$MAX_SAMPLES_PER_GROUP" \
      --num_epochs "$KNOWLEDGE_EPOCHS" \
      --train_micro_batch_size "$KNOWLEDGE_MICRO_BATCH_SIZE" \
      --dynamic_train_padding \
      --gradient_checkpointing \
      --learning_rate "$LEARNING_RATE" \
      --lora_weight_learning_rate "$LORA_WEIGHT_LEARNING_RATE" \
      --num_checkpoints "$NUM_CHECKPOINTS" \
      --mode train_only \
      2>&1 | tee "$KNOWLEDGE_DIR/run.log"
  fi
}

run_evaluation() {
  if [[ ! -d "$TASK_DIR/final/task_shared" || ! -d "$KNOWLEDGE_DIR/final" ]]; then
    echo "Training outputs are missing under $OUTPUT_ROOT" >&2
    exit 1
  fi

  for eval_seed in 42 43 44 45 46; do
    users_split="data_splits/wvs/test_users_seed${eval_seed}.json"
    eval_dir="$OUTPUT_ROOT/main/seed_${eval_seed}"
    if [[ -f "$eval_dir/eval_results.json" ]]; then
      echo "Seed $eval_seed already complete: $eval_dir/eval_results.json"
      continue
    fi
    mkdir -p "$eval_dir"

    EVAL_ARGS=(
      "${COMMON_ARGS[@]}"
      --users_split "$users_split"
      --output_dir "$eval_dir"
      --lora_weight_mode residual_gate
      --lora_weight_normalize sum_to_active_count
      --lora_composition_mode explicit_task_knowledge_projection
      --lora_orthogonal_strength 0
      --lora_soft_orthogonal_lambda "$SOFT_LAMBDA"
      --lora_task_scale "$INFERENCE_TASK_SCALE"
      --lora_knowledge_scale "$INFERENCE_KNOWLEDGE_SCALE"
      --lora_train_stage knowledge_only
      --load_task_lora_dir "$TASK_DIR/final"
      --freeze_task_lora
      --load_lora_dir "$KNOWLEDGE_DIR/final"
      --eval_batch_size "$EVAL_BATCH_SIZE"
      --max_eval_users "$MAX_EVAL_USERS"
      --eval_exp3_only
      --direct_answer_eval
      --invalid_prediction_fallback choice_logprob
      --max_new_tokens "$MAX_NEW_TOKENS"
      --temperature 0
      --mode eval_only
    )
    if [[ "$USE_VLLM_EVAL" == "1" ]]; then
      EVAL_ARGS+=(
        --use_vllm_eval
        --vllm_dp_size "$NPROC"
        --vllm_tp_size 1
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --vllm_max_model_len 4096
      )
    fi

    echo "Evaluating WVS seed $eval_seed"
    "$PYTHON_BIN" src/train_eval.py "${EVAL_ARGS[@]}" \
      2>&1 | tee "$eval_dir/run.log"
  done
}

echo "Model=$MODEL_KEY lambda=$SOFT_LAMBDA rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "Inference scales: task=$INFERENCE_TASK_SCALE knowledge=$INFERENCE_KNOWLEDGE_SCALE"
echo "LoRA targets: $TARGET_MODULES_STR"
echo "Outputs: $OUTPUT_ROOT"

if [[ "$ACTION" == "all" || "$ACTION" == "train" ]]; then
  run_training
fi
if [[ "$ACTION" == "all" || "$ACTION" == "eval" ]]; then
  run_evaluation
fi
