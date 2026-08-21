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
: "${TASK_ANSWER_MODE:?TASK_ANSWER_MODE is required}"
: "${TASK_BALANCE_ANSWERS:?TASK_BALANCE_ANSWERS is required}"
: "${TASK_MICRO_BATCH_SIZE:?TASK_MICRO_BATCH_SIZE is required}"
: "${KNOWLEDGE_MICRO_BATCH_SIZE:?KNOWLEDGE_MICRO_BATCH_SIZE is required}"
: "${KNOWLEDGE_GRAD_ACCUM:?KNOWLEDGE_GRAD_ACCUM is required}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/sociobench/$MODEL_KEY"}
SOCIOBENCH_ROOT=${SOCIOBENCH_ROOT:-"data/SocioBench"}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-"data_splits/sociobench/lambda_validation.json"}

LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
TARGET_MODULES_STR=${TARGET_MODULES_STR:-"gate_proj up_proj down_proj q_proj k_proj v_proj o_proj"}
FEATURE_DIMENSIONS_STR=${FEATURE_DIMENSIONS_STR:-"gender age_group country religion education marital_status employment urban_rural"}
TASK_MAX_SAMPLES=${TASK_MAX_SAMPLES:-24000}
MAX_TRAIN_RESPONDENTS_PER_DOMAIN=${MAX_TRAIN_RESPONDENTS_PER_DOMAIN:-160}
MAX_SAMPLES_PER_GROUP=${MAX_SAMPLES_PER_GROUP:-32}
TASK_EPOCHS=${TASK_EPOCHS:-1}
KNOWLEDGE_EPOCHS=${KNOWLEDGE_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
TASK_GRAD_ACCUM=${TASK_GRAD_ACCUM:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_WEIGHT_LEARNING_RATE=${LORA_WEIGHT_LEARNING_RATE:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29610}

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

for required_path in "$MODEL_PATH" "$SOCIOBENCH_ROOT" "$TRAIN_MANIFEST"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --sociobench_root "$SOCIOBENCH_ROOT"
  --model_name "$MODEL_PATH"
  --domains all
  --dataset_size 500
  --train_ratio 0.8
  --question_split_ratio 0.8
  --min_required_feature_dimensions 8
  --max_train_respondents_per_domain "$MAX_TRAIN_RESPONDENTS_PER_DOMAIN"
  --feature_dimensions "${FEATURE_DIMENSIONS[@]}"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --lora_target_modules "${TARGET_MODULES[@]}"
  --lora_dimension_order "${FEATURE_DIMENSIONS[@]}"
  --task_lora_name task_shared
  --task_answer_mode "$TASK_ANSWER_MODE"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --lora_weight_learning_rate "$LORA_WEIGHT_LEARNING_RATE"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --max_length 1024
  --train_answer_value_only
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --max_eval_respondents_per_domain 0
  --max_new_tokens "$MAX_NEW_TOKENS"
  --eval_decoding generate
  --invalid_prediction_fallback choice_logprob
  --temperature 0
  --num_checkpoints "$NUM_CHECKPOINTS"
  --seed 42
)

TASK_BALANCE_ARGS=()
if [[ "$TASK_BALANCE_ANSWERS" == "1" ]]; then
  TASK_BALANCE_ARGS+=(--task_balance_answers)
fi

run_training() {
  mkdir -p "$TASK_DIR" "$KNOWLEDGE_DIR"

  if [[ -d "$TASK_DIR/final/task_shared" ]]; then
    echo "Stage I already complete: $TASK_DIR/final/task_shared"
  else
    echo "Stage I: training the demographic-agnostic task LoRA"
    "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT_BASE" \
      sociobench_code/train_eval.py \
      "${COMMON_ARGS[@]}" \
      --split_manifest "$TRAIN_MANIFEST" \
      --output_dir "$TASK_DIR" \
      --lora_weight_mode residual_gate \
      --lora_weight_normalize sum_to_active_count \
      --lora_composition_mode explicit_task_knowledge_projection \
      --lora_orthogonal_strength 0 \
      --lora_soft_orthogonal_lambda 0 \
      --lora_soft_orthogonal_mode a_frobenius \
      --lora_task_scale 1 \
      --lora_knowledge_scale 1 \
      --lora_train_stage task_only \
      --task_train_data_mode grouped \
      "${TASK_BALANCE_ARGS[@]}" \
      --task_max_samples "$TASK_MAX_SAMPLES" \
      --max_samples_per_group -1 \
      --num_epochs "$TASK_EPOCHS" \
      --train_micro_batch_size "$TASK_MICRO_BATCH_SIZE" \
      --gradient_accumulation_steps "$TASK_GRAD_ACCUM" \
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
      sociobench_code/train_eval.py \
      "${COMMON_ARGS[@]}" \
      --split_manifest "$TRAIN_MANIFEST" \
      --output_dir "$KNOWLEDGE_DIR" \
      --lora_weight_mode residual_gate \
      --lora_weight_normalize sum_to_active_count \
      --lora_composition_mode explicit_task_knowledge_projection \
      --lora_orthogonal_strength 0 \
      --lora_soft_orthogonal_lambda "$SOFT_LAMBDA" \
      --lora_soft_orthogonal_mode a_frobenius \
      --lora_task_scale 1 \
      --lora_knowledge_scale 1 \
      --lora_train_stage knowledge_only \
      --load_task_lora_dir "$TASK_DIR/final" \
      --freeze_task_lora \
      --max_samples_per_group "$MAX_SAMPLES_PER_GROUP" \
      --num_epochs "$KNOWLEDGE_EPOCHS" \
      --train_micro_batch_size "$KNOWLEDGE_MICRO_BATCH_SIZE" \
      --gradient_accumulation_steps "$KNOWLEDGE_GRAD_ACCUM" \
      --dynamic_train_padding \
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
    manifest="data_splits/sociobench/test_users_seed${eval_seed}.json"
    eval_dir="$OUTPUT_ROOT/main/seed_${eval_seed}"
    if [[ -f "$eval_dir/eval_results.json" ]]; then
      echo "Seed $eval_seed already complete: $eval_dir/eval_results.json"
      continue
    fi
    mkdir -p "$eval_dir"

    echo "Evaluating SocioBench seed $eval_seed"
    "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" \
      --master_port="$((MASTER_PORT_BASE + 10 + eval_seed - 42))" \
      sociobench_code/train_eval.py \
      "${COMMON_ARGS[@]}" \
      --split_manifest "$manifest" \
      --eval_include_person_ids_manifest "$manifest" \
      --output_dir "$eval_dir" \
      --lora_weight_mode residual_gate \
      --lora_weight_normalize sum_to_active_count \
      --lora_composition_mode explicit_task_knowledge_projection \
      --lora_orthogonal_strength 0 \
      --lora_soft_orthogonal_lambda "$SOFT_LAMBDA" \
      --lora_soft_orthogonal_mode a_frobenius \
      --lora_task_scale "$INFERENCE_TASK_SCALE" \
      --lora_knowledge_scale "$INFERENCE_KNOWLEDGE_SCALE" \
      --lora_train_stage knowledge_only \
      --load_task_lora_dir "$TASK_DIR/final" \
      --freeze_task_lora \
      --load_lora_dir "$KNOWLEDGE_DIR/final" \
      --eval_exp3_only \
      --save_prediction_records \
      --mode eval_only \
      2>&1 | tee "$eval_dir/run.log"
  done
}

echo "Model=$MODEL_KEY lambda=$SOFT_LAMBDA rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "Stage-I answers: $TASK_ANSWER_MODE (balance=$TASK_BALANCE_ANSWERS)"
echo "Inference scales: task=$INFERENCE_TASK_SCALE knowledge=$INFERENCE_KNOWLEDGE_SCALE"
echo "LoRA targets: $TARGET_MODULES_STR"
echo "Outputs: $OUTPUT_ROOT"

if [[ "$ACTION" == "all" || "$ACTION" == "train" ]]; then
  run_training
fi
if [[ "$ACTION" == "all" || "$ACTION" == "eval" ]]; then
  run_evaluation
fi
