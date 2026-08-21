"""SocioBench training and evaluation entry point for PARADE.

The locked experiment reads respondent/question partitions from a split
manifest and harmonizes SocioBench attributes to the same eight dimensions as
WVS: gender, age group, country, religion, education, marital status,
employment status, and urban/rural residence.  Only complete eight-value
profiles enter the main runs.

The ten ISSP domains share one Task LoRA and one pool of attribute-value
Demographic LoRAs.  In every main run, Stage 1 learns the common answer
protocol from balanced random legal option identifiers.  Stage 2 freezes that
module and co-trains the active demographic modules with dimension-level
weights and soft orthogonality.  Evaluation is performed per domain using
the manifest's held-out respondents and questions.
"""

import os
import sys
import json
import argparse
import gc
import time
import types
import importlib.machinery
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, OrderedDict
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _runtime_hardware_metadata() -> dict:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": props.name,
                "memory_gib": round(props.total_memory / (1024 ** 3), 2),
            })
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def _install_sklearn_stub_if_needed():
    """Avoid broken optional sklearn imports in transformers on this env."""
    if "sklearn.metrics" in sys.modules:
        return

    sklearn_mod = types.ModuleType("sklearn")
    metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_mod.__spec__ = importlib.machinery.ModuleSpec(
        "sklearn.metrics", loader=None)

    def roc_curve(*args, **kwargs):
        raise ImportError(
            "sklearn is unavailable in this environment; roc_curve is optional "
            "for this SocioBench training script.")

    metrics_mod.roc_curve = roc_curve
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod


_install_sklearn_stub_if_needed()

# Add current project and src/ to path for shared modules.
_code_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_code_dir)
_src_dir = os.path.join(_project_root, "src")
for _path in (_src_dir, _project_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lora import MultiLoRAModelWrapper
from anonymization import public_path, sanitize_for_publication
from dataloader import QADataset
from utils import generate_persona_prompt, extract_answer_number
from train_eval import (
    train_multi_lora as current_train_multi_lora,
    _load_task_lora,
)
from sociobench_code.utils import (
    SOCIOBENCH_DOMAINS, COUNTRY_ATTR_KEYS,
    get_sociobench_user_features, load_sociobench_qa,
    load_sociobench_ground_truth,
    is_invalid_answer, get_question_country_code,
    generate_sociobench_persona_prompt,
)
from sociobench_code.option_metrics import (
    normalized_option_distance,
    ordinal_core,
)
from distribution_metrics import compute_distribution_metrics
from model_loading import (
    load_backbone_config,
    load_backbone_model,
    load_backbone_tokenizer,
)

DEFAULT_FEATURE_DIMENSIONS = [
    "gender", "age_group", "country", "religion", "education",
    "marital_status", "employment", "urban_rural",
]


# ======================================================================
# Distributed Setup
# ======================================================================

def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def format_eval_prompts(tokenizer, prompts: List[str]) -> List[str]:
    """Apply chat template while keeping all instructions in the user prompt."""
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        return prompts

    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    return formatted


def build_sociobench_eval_prompt(
    question_text: str,
    opts_fmt: str,
    persona_prompt: str = "",
    *,
    answer_value_only: bool = False,
) -> str:
    """Build an evaluation prompt aligned with the WVS output contract."""
    profile_prefix = f"{persona_prompt}\n\n" if persona_prompt else ""
    if answer_value_only:
        return (
            f"{profile_prefix}"
            f"Question: {question_text}\n"
            f"Options:\n{opts_fmt}\n\n"
            f"Write ONLY your chosen option number.\n\n"
            f"Answer:"
        )

    if persona_prompt:
        return (
            f"{profile_prefix}"
            f"Question: {question_text}\n"
            f"Options:\n{opts_fmt}\n\n"
            f"Please answer this question based on your demographic background. "
            f"First, provide your reasoning and explanation. "
            f"Then, on a new line, write ONLY your chosen option number "
            f"inside <answer></answer> tags.\n\n"
            f"Required format:\n"
            f"[Your reasoning and explanation]\n"
            f"<answer>[option number]</answer>\n\n"
            f"Your response:"
        )

    return (
        f"Question: {question_text}\n"
        f"Options:\n{opts_fmt}\n\n"
        f"First, provide your reasoning and explanation. "
        f"Then, on a new line, write ONLY your chosen option number "
        f"inside <answer></answer> tags.\n\n"
        f"Required format:\n"
        f"[Your reasoning and explanation]\n"
        f"<answer>[option number]</answer>\n\n"
        f"Your response:"
    )


def _macro_micro_f1(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
    if not y_true or not y_pred:
        return 0.0, 0.0
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    )
    micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) else 0.0
    )
    return macro, micro


def _subgroup_value(attrs: dict, domain_id: int, group: str) -> str:
    if group == "country":
        key = COUNTRY_ATTR_KEYS.get(domain_id, "")
        return str(attrs.get(key, "")).strip() or "unknown"
    if group == "gender":
        return (
            str(attrs.get("Sex of Respondent")
                or attrs.get("Sex of respondent")
                or "unknown").strip()
        )
    if group == "age":
        try:
            age = int(float(str(attrs.get("Age of respondent", ""))))
            if age <= 30:
                return "young"
            if age <= 55:
                return "middle"
            return "senior"
        except (ValueError, TypeError):
            return "unknown"
    if group == "employment":
        return str(
            attrs.get("Main status")
            or attrs.get("Currently, formerly, or never in paid work")
            or attrs.get("Employment relationship")
            or "unknown"
        ).strip()
    if group == "occupation":
        for key in attrs:
            if key.startswith("Occupation ISCO"):
                return str(attrs.get(key, "unknown")).strip()
        return "unknown"
    return "unknown"


def _new_metric_bucket():
    return {
        "correct": 0,
        "total": 0,
        "invalid_predictions": 0,
        "valid_count": 0,
        "error_sum": 0.0,
        "y_true": [],
        "y_pred": [],
    }


def _update_metric_bucket(bucket: dict, gt: int, pred: Optional[int],
                          valid_options: Optional[frozenset]):
    bucket["total"] += 1
    if pred is None:
        bucket["invalid_predictions"] += 1
        return
    bucket["valid_count"] += 1
    bucket["y_true"].append(str(gt))
    bucket["y_pred"].append(str(pred))
    if pred == gt:
        bucket["correct"] += 1
    bucket["error_sum"] += normalized_option_distance(
        pred, gt, valid_options or (gt,)
    )


def _merge_metric_buckets(left: dict, right: dict):
    left["correct"] += right.get("correct", 0)
    left["total"] += right.get("total", 0)
    left["invalid_predictions"] += right.get("invalid_predictions", 0)
    left["valid_count"] += right.get("valid_count", 0)
    left["error_sum"] += right.get("error_sum", 0.0)
    left["y_true"].extend(right.get("y_true", []))
    left["y_pred"].extend(right.get("y_pred", []))


def _finalize_metric_bucket(bucket: dict) -> dict:
    total = bucket.get("total", 0)
    valid = bucket.get("valid_count", 0)
    macro_f1, micro_f1 = _macro_micro_f1(
        bucket.get("y_true", []), bucket.get("y_pred", []))
    return {
        "accuracy": bucket.get("correct", 0) / total if total else 0.0,
        "correct": bucket.get("correct", 0),
        "total": total,
        "invalid_predictions": bucket.get("invalid_predictions", 0),
        "valid_count": valid,
        "valid_rate": valid / total if total else 0.0,
        "mae": bucket.get("error_sum", 0.0) / valid if valid else 0.0,
        "option_distance": bucket.get("error_sum", 0.0) / valid if valid else 0.0,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


def _finalize_group_metrics(group_metrics: dict) -> dict:
    finalized = {}
    for group_name, buckets in group_metrics.items():
        finalized[group_name] = {
            str(key): _finalize_metric_bucket(bucket)
            for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0]))
        }
    return finalized


# ======================================================================
# Data Split
# ======================================================================

def split_questions(qa_data: List[dict], train_ratio: float = 0.8,
                    seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Split QA questions into train/test sets."""
    rng = random.Random(seed)
    shuffled = qa_data[:]
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]


def split_respondents(ground_truth_list: List[dict], domain_id: int,
                      train_ratio: float = 0.8, seed: int = 42,
                      feature_dimensions: Optional[List[str]] = None,
                      min_required_feature_dimensions: int = 3,
                      ) -> Tuple[List[dict], List[dict]]:
    """Split respondents into train/test with stratification by feature group."""
    rng = random.Random(seed)
    feature_dimensions = list(feature_dimensions or DEFAULT_FEATURE_DIMENSIONS)
    required_features = min(
        len(feature_dimensions), max(1, min_required_feature_dimensions))

    # Group by features
    groups: Dict[frozenset, List[dict]] = defaultdict(list)
    no_features = []
    for r in ground_truth_list:
        features = get_sociobench_user_features(
            r.get("attributes", {}), domain_id, feature_dimensions)
        if features and len(features) >= required_features:
            groups[features].append(r)
        else:
            no_features.append(r)

    train, test = [], []
    for features, respondents in groups.items():
        rng.shuffle(respondents)
        n_train = max(1, int(len(respondents) * train_ratio))
        train.extend(respondents[:n_train])
        test.extend(respondents[n_train:])

    # Respondents without features go to test (can't train on them)
    test.extend(no_features)

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ======================================================================
# Build Training Data
# ======================================================================

def build_sociobench_training_groups(
    train_respondents: List[dict],
    qa_data: List[dict],
    domain_id: int,
    domain_name: str,
    feature_dimensions: Optional[List[str]] = None,
    min_required_feature_dimensions: int = 3,
) -> Dict[frozenset, List[dict]]:
    """Build QA pairs grouped by demographic features from SocioBench data.

    Returns dict[frozenset[str], list[dict]] compatible with QADataset.
    """
    # Build question map
    qa_map = {}
    for q in qa_data:
        qid = q.get("question_id", "")
        if qid:
            qa_map[qid.lower()] = q

    groups: Dict[frozenset, List[dict]] = defaultdict(list)
    skipped_no_features = 0
    skipped_invalid = 0
    feature_dimensions = list(feature_dimensions or DEFAULT_FEATURE_DIMENSIONS)
    required_features = min(
        len(feature_dimensions), max(1, min_required_feature_dimensions))

    for respondent in train_respondents:
        attrs = respondent.get("attributes", {})
        features = get_sociobench_user_features(
            attrs, domain_id, feature_dimensions)
        if not features or len(features) < required_features:
            skipped_no_features += 1
            continue

        # Get country info for this respondent
        country_key = COUNTRY_ATTR_KEYS.get(domain_id, "")
        country_name = str(attrs.get(country_key, "")).strip()

        qa_pairs = respondent.get("questions_answer", {})
        for qid_raw, true_answer in qa_pairs.items():
            qid = str(qid_raw)

            if true_answer is None or is_invalid_answer(true_answer):
                skipped_invalid += 1
                continue
            true_answer_str = str(true_answer).strip()
            if not true_answer_str:
                continue

            # Check country-specific questions
            q_country = get_question_country_code(qid)
            if q_country and country_name:
                pass  # Keep all for training

            qa_item = qa_map.get(qid.lower())
            if not qa_item:
                continue

            question_text = qa_item.get("question", "")
            options = dict(qa_item.get("answer", {}))
            special = qa_item.get("special", {})
            if isinstance(special, dict):
                for code, overrides in special.items():
                    if isinstance(overrides, dict):
                        options.update(overrides)
                        break

            if not question_text or not options:
                continue

            try:
                gt_val = int(float(true_answer_str))
            except (ValueError, TypeError):
                skipped_invalid += 1
                continue

            # Check gt_val is a valid option
            int_keys = [int(k) for k in options.keys() if k.lstrip("-").isdigit()]
            if int_keys and gt_val not in int_keys:
                skipped_invalid += 1
                continue

            answer_text = options.get(str(gt_val), f"Option {gt_val}")

            opts_fmt = "\n".join(
                f"{k}. {v}" for k, v in
                sorted(options.items(),
                       key=lambda x: int(x[0]) if x[0].lstrip("-").isdigit() else 0)
            )

            groups[features].append({
                "question": f"{question_text}\nOptions:\n{opts_fmt}",
                "answer": answer_text,
                "answer_value": str(gt_val),
                "valid_answer_values": [
                    str(value) for value in sorted(int_keys) if value >= 0
                ],
                "answer_text_by_value": {
                    str(value): options[str(value)]
                    for value in int_keys
                    if value >= 0 and str(value) in options
                },
            })

    total = sum(len(v) for v in groups.values())
    print(f"  [{domain_name}] {total} QA pairs in {len(groups)} feature groups "
          f"(skipped: {skipped_no_features} no-features, {skipped_invalid} invalid)")
    return dict(groups)


def merge_groups(all_groups: List[Dict[frozenset, List[dict]]]
                 ) -> Dict[frozenset, List[dict]]:
    """Merge multiple group dicts into one."""
    merged: Dict[frozenset, List[dict]] = defaultdict(list)
    for groups in all_groups:
        for features, items in groups.items():
            merged[features].extend(items)
    return dict(merged)


def collapse_groups_to_task_pool(groups: Dict[frozenset, List[dict]],
                                 args) -> Dict[frozenset, List[dict]]:
    """Use all SocioBench QA examples as one demographic-agnostic task pool."""
    rng = random.Random(args.seed)
    items = []
    for qa_list in groups.values():
        items.extend(dict(item) for item in qa_list)
    answer_mode = getattr(args, "task_answer_mode", "real")
    if answer_mode == "balanced_random":
        balanced_items = []
        for item in items:
            valid_values = list(item.get("valid_answer_values", []))
            answer_text = item.get("answer_text_by_value", {})
            if not valid_values:
                continue
            sampled_value = rng.choice(valid_values)
            sampled_text = answer_text.get(sampled_value)
            if sampled_text is None:
                continue
            item["answer_value"] = sampled_value
            item["answer"] = sampled_text
            balanced_items.append(item)
        items = balanced_items
    if getattr(args, "task_balance_answers", False):
        by_answer = defaultdict(list)
        for item in items:
            by_answer[str(item.get("answer_value", ""))].append(item)
        for bucket in by_answer.values():
            rng.shuffle(bucket)
        if by_answer:
            if args.task_max_samples > 0:
                per_answer = max(1, args.task_max_samples // len(by_answer))
                balanced = []
                leftovers = []
                for bucket in by_answer.values():
                    balanced.extend(bucket[:per_answer])
                    leftovers.extend(bucket[per_answer:])
                rng.shuffle(leftovers)
                balanced.extend(
                    leftovers[:max(0, args.task_max_samples - len(balanced))]
                )
                items = balanced[:args.task_max_samples]
            else:
                per_answer = min(len(bucket) for bucket in by_answer.values())
                items = [
                    item
                    for bucket in by_answer.values()
                    for item in bucket[:per_answer]
                ]
    elif args.task_max_samples > 0 and len(items) > args.task_max_samples:
        items = rng.sample(items, args.task_max_samples)
    rng.shuffle(items)
    print(f"Task pool: {len(items):,} QA pairs -> {args.task_lora_name} "
          f"(answer_mode={answer_mode}, balance="
          f"{bool(getattr(args, 'task_balance_answers', False))})")
    return {frozenset({args.task_lora_name}): items}


# ======================================================================
# Training (adapted from src/train_eval.py)
# ======================================================================

def _collate(batch):
    return {
        k: torch.tensor([ex[k] for ex in batch], dtype=torch.long)
        for k in ('input_ids', 'labels', 'attention_mask')
    }


def train_multi_lora(wrapper: MultiLoRAModelWrapper, groups: dict,
                     tokenizer, args, rank: int = 0, world_size: int = 1,
                     local_rank: int = 0):
    device = torch.device(f"cuda:{local_rank}")
    collate_fn = _collate

    if is_main_process(rank):
        print("\nTokenizing training data ...")
    group_datasets: Dict[frozenset, QADataset] = {}
    for feat_key, qa_list in tqdm(groups.items(), desc="Tokenize",
                                   disable=not is_main_process(rank)):
        group_datasets[feat_key] = QADataset(qa_list, tokenizer, args.max_length)
    total_samples = sum(len(ds) for ds in group_datasets.values())
    if is_main_process(rank):
        print(f"Total tokenized samples: {total_samples:,}")

    use_ddp = world_size > 1

    optimizer = torch.optim.AdamW(
        wrapper.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    wrapper.base_model.train()

    keys = list(group_datasets.keys())
    loaders = {}
    for feat_key in keys:
        ds = group_datasets[feat_key]
        if len(ds) > 0:
            if use_ddp:
                sampler = DistributedSampler(ds, num_replicas=world_size,
                                             rank=rank, shuffle=True)
                loaders[feat_key] = DataLoader(
                    ds, batch_size=args.batch_size, sampler=sampler,
                    collate_fn=collate_fn, drop_last=False,
                    num_workers=2, pin_memory=True)
            else:
                loaders[feat_key] = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False,
                    num_workers=2, pin_memory=True)

    total_batches = sum(len(loader) for loader in loaders.values())
    valid_keys = list(loaders.keys())

    # Use a synchronized RNG for feature group selection across ranks
    sync_rng = random.Random(args.seed)

    for epoch in range(args.num_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        # Reset sync RNG each epoch so all ranks pick same sequence
        sync_rng = random.Random(args.seed + epoch)

        if use_ddp:
            for feat_key in valid_keys:
                loaders[feat_key].sampler.set_epoch(epoch)

        loader_iters = {k: iter(loaders[k]) for k in valid_keys}

        pbar = tqdm(range(total_batches),
                    desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                    disable=not is_main_process(rank))
        for _ in pbar:
            # All ranks pick the same feature group
            feat_key = sync_rng.choice(valid_keys)
            try:
                batch = next(loader_iters[feat_key])
            except StopIteration:
                loader_iters[feat_key] = iter(loaders[feat_key])
                batch = next(loader_iters[feat_key])

            batch = {k: v.to(device) for k, v in batch.items()}

            try:
                wrapper.set_active_loras(feat_key)
                optimizer.zero_grad()
                out = wrapper.base_model(**batch)
                loss = out.loss
                loss.backward()

                # Manual gradient sync: average gradients across ranks
                if use_ddp:
                    for p in wrapper.trainable_parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                torch.nn.utils.clip_grad_norm_(
                    wrapper.trainable_parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_steps += 1
            except Exception as e:
                if is_main_process(rank):
                    print(f"\nWarning: Error in training step: {e}")
                continue

            if is_main_process(rank):
                pbar.set_postfix(loss=f"{epoch_loss / max(epoch_steps, 1):.4f}")

        elapsed = time.time() - t0
        avg = epoch_loss / max(epoch_steps, 1)
        if is_main_process(rank):
            print(f"Epoch {epoch + 1} | {elapsed:.0f}s | "
                  f"avg loss {avg:.4f} | steps {epoch_steps}")

        # Only rank 0 saves checkpoints
        if is_main_process(rank):
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            wrapper.save_all_loras(ckpt_dir)

        if use_ddp:
            dist.barrier()

    if is_main_process(rank):
        wrapper.save_all_loras(os.path.join(args.output_dir, "final"))
        print("\nTraining complete. Adapters -> "
              f"{public_path(args.output_dir)}")

    if use_ddp:
        dist.barrier()


# ======================================================================
# Evaluation (adapted from src/sociobench_eval.py)
# ======================================================================

def _select_numeric_option_by_risk(option_scores, args) -> int:
    """Select an option by zero-one risk plus normalized ordinal distance."""
    options = sorted(option_scores)
    if (args.eval_decoding != "choice_risk"
            or args.eval_mae_risk_weight <= 0):
        return max(options, key=option_scores.__getitem__)

    temperature = max(float(args.eval_choice_temperature), 1e-6)
    scores = torch.tensor(
        [option_scores[option] for option in options], dtype=torch.float64)
    probabilities = torch.softmax(scores / temperature, dim=0)
    risks = []
    for prediction_index, prediction in enumerate(options):
        classification_risk = 1.0 - float(probabilities[prediction_index])
        ordinal_risk = sum(
            float(probability)
            * normalized_option_distance(prediction, target, options)
            for target, probability in zip(options, probabilities)
        )
        risks.append(
            classification_risk
            + args.eval_mae_risk_weight * ordinal_risk
        )
    return options[min(range(len(options)), key=risks.__getitem__)]

def _score_numeric_options(
    wrapper: MultiLoRAModelWrapper,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_options_list: List[Optional[frozenset]],
    args,
    *,
    device: torch.device,
) -> List[Optional[int]]:
    """Choose among numeric labels, including multi-token labels such as -4."""
    next_token_logits = wrapper.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits[:, -1, :]
    predictions: List[Optional[int]] = []
    for row, valid_options in enumerate(valid_options_list):
        if not valid_options:
            predictions.append(None)
            continue
        option_tokens = {
            option: tokenizer.encode(str(option), add_special_tokens=False)
            for option in sorted(valid_options)
        }
        if all(len(tokens) == 1 for tokens in option_tokens.values()):
            option_scores = {
                option: float(next_token_logits[
                    row, tokens[0]
                ].detach().cpu())
                for option, tokens in option_tokens.items()
            }
            prediction = _select_numeric_option_by_risk(option_scores, args)
            predictions.append(prediction)
            continue

        prompt_ids = input_ids[row][attention_mask[row].bool()].tolist()
        candidate_options = list(option_tokens)
        candidate_ids = [
            prompt_ids + option_tokens[option]
            for option in candidate_options
        ]
        max_candidate_length = max(len(ids) for ids in candidate_ids)
        pad_id = (tokenizer.pad_token_id
                  if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)
        candidate_input_ids = torch.full(
            (len(candidate_ids), max_candidate_length),
            pad_id, dtype=torch.long, device=device)
        candidate_attention = torch.zeros_like(candidate_input_ids)
        for candidate_index, ids in enumerate(candidate_ids):
            length = len(ids)
            candidate_input_ids[candidate_index, :length] = torch.tensor(
                ids, device=device)
            candidate_attention[candidate_index, :length] = 1
        candidate_logits = wrapper.base_model(
            input_ids=candidate_input_ids,
            attention_mask=candidate_attention,
        ).logits
        candidate_scores = []
        prompt_length = len(prompt_ids)
        for candidate_index, option in enumerate(candidate_options):
            tokens = option_tokens[option]
            positions = torch.arange(
                prompt_length - 1,
                prompt_length - 1 + len(tokens),
                device=device)
            token_tensor = torch.tensor(tokens, device=device)
            token_log_probs = candidate_logits[
                candidate_index, positions
            ].log_softmax(dim=-1).gather(
                1, token_tensor.unsqueeze(1)
            ).squeeze(1)
            candidate_scores.append(
                float(token_log_probs.mean().detach().cpu()))
        predictions.append(_select_numeric_option_by_risk(
            dict(zip(candidate_options, candidate_scores)), args))
    return predictions

def evaluate_sociobench_domain(
    wrapper: MultiLoRAModelWrapper,
    qa_data: List[dict],
    test_respondents: List[dict],
    domain_name: str,
    domain_id: int,
    tokenizer,
    args,
    *,
    desc: str = "Eval",
    use_lora: bool = True,
    use_persona_prompt: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> dict:
    """Evaluate on test respondents of a single SocioBench domain.
    Multi-GPU: shards feature groups across ranks, then aggregates."""
    wrapper.base_model.eval()
    device = torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    input_token_total = 0
    output_token_total = 0

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    qa_map = {}
    for q in qa_data:
        qid = q.get("question_id", "")
        if qid:
            qa_map[qid.lower()] = q

    feature_dimensions = list(
        getattr(args, "feature_dimensions", None) or DEFAULT_FEATURE_DIMENSIONS)
    required_features = min(
        len(feature_dimensions),
        max(1, getattr(args, "min_required_feature_dimensions", 3)))
    user_groups: Dict[frozenset, List[Tuple[dict, dict]]] = defaultdict(list)
    skipped = 0
    for respondent in test_respondents:
        attrs = respondent.get("attributes", {})
        features = get_sociobench_user_features(
            attrs, domain_id, feature_dimensions)
        if not features or len(features) < required_features:
            skipped += 1
            continue
        user_groups[features].append((respondent, attrs))

    if is_main_process(rank):
        print(f"  {len(user_groups)} feature groups, {skipped} skipped")

    # Shard feature groups across ranks
    all_group_keys = sorted(user_groups.keys())
    my_group_keys = all_group_keys[rank::world_size]

    correct = 0
    total = 0
    invalid_predictions = 0
    error_sum = 0.0
    y_true_all = []
    y_pred_all = []
    distribution_records = []
    sample_outputs = []
    group_metrics = {
        "country_metrics": defaultdict(_new_metric_bucket),
        "gender_metrics": defaultdict(_new_metric_bucket),
        "age_metrics": defaultdict(_new_metric_bucket),
        "occupation_metrics": defaultdict(_new_metric_bucket),
        "employment_metrics": defaultdict(_new_metric_bucket),
    }
    feature_dimensions = list(
        getattr(args, "feature_dimensions", None) or DEFAULT_FEATURE_DIMENSIONS)
    required_features = min(
        len(feature_dimensions),
        max(1, getattr(args, "min_required_feature_dimensions", 3)))

    try:
        with torch.no_grad():
            for features in tqdm(my_group_keys, desc=desc,
                                  disable=not is_main_process(rank)):
                respondents = user_groups[features]
                if not use_lora:
                    wrapper.set_active_loras(set())
                elif args.eval_lora_components == "task_only":
                    wrapper.set_active_loras([args.task_lora_name])
                elif args.eval_lora_components == "knowledge_only":
                    wrapper.set_active_loras_exact(features)
                else:
                    wrapper.set_active_loras(features)

                country_key = COUNTRY_ATTR_KEYS.get(domain_id, "")

                prompts, gts, qids, valid_opts_list, meta_list = [], [], [], [], []

                for respondent, attrs in respondents:
                    qa_pairs = respondent.get("questions_answer", {})
                    r_country_name = str(attrs.get(country_key, "")).strip()
                    persona_prompt = ""
                    if use_persona_prompt:
                        persona_prompt = generate_sociobench_persona_prompt(
                            attrs, domain_id, feature_dimensions)
                        if not persona_prompt:
                            persona_prompt = generate_persona_prompt(features)

                    for qid_raw, true_answer in qa_pairs.items():
                        qid = str(qid_raw)
                        if true_answer is None or is_invalid_answer(true_answer):
                            continue
                        true_answer_str = str(true_answer).strip()
                        if not true_answer_str:
                            continue

                        qa_item = qa_map.get(qid.lower())
                        if not qa_item:
                            continue

                        question_text = qa_item.get("question", "")
                        options = dict(qa_item.get("answer", {}))
                        if not question_text or not options:
                            continue

                        try:
                            gt_val = int(float(true_answer_str))
                        except (ValueError, TypeError):
                            continue

                        int_keys = [int(k) for k in options.keys()
                                    if k.lstrip("-").isdigit()]
                        valid_options = frozenset(int_keys) if int_keys else None
                        if valid_options and gt_val not in valid_options:
                            continue

                        opts_fmt = "\n".join(
                            f"{k}. {v}" for k, v in
                            sorted(options.items(),
                                   key=lambda x: int(x[0]) if x[0].lstrip("-").isdigit() else 0)
                        )

                        prompt = build_sociobench_eval_prompt(
                            question_text,
                            opts_fmt,
                            persona_prompt if use_persona_prompt else "",
                            answer_value_only=args.train_answer_value_only,
                        )

                        prompts.append(prompt)
                        gts.append(gt_val)
                        qids.append(qid)
                        valid_opts_list.append(valid_options)
                        meta_list.append({
                            "person_id": str(respondent.get("person_id", "")),
                            "country": _subgroup_value(attrs, domain_id, "country"),
                            "gender": _subgroup_value(attrs, domain_id, "gender"),
                            "age": _subgroup_value(attrs, domain_id, "age"),
                            "occupation": _subgroup_value(attrs, domain_id, "occupation"),
                            "employment": _subgroup_value(attrs, domain_id, "employment"),
                        })

                if not prompts:
                    continue

                for i in range(0, len(prompts), args.eval_batch_size):
                    bp = prompts[i:i + args.eval_batch_size]
                    bg = gts[i:i + args.eval_batch_size]
                    bv = valid_opts_list[i:i + args.eval_batch_size]
                    bm = meta_list[i:i + args.eval_batch_size]
                    model_prompts = format_eval_prompts(tokenizer, bp)

                    try:
                        enc = tokenizer(
                            model_prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_length
                        )
                        input_ids = enc["input_ids"].to(device)
                        attn_mask = enc["attention_mask"].to(device)
                        input_token_total += int(attn_mask.sum().item())

                        responses = []
                        predictions = []
                        if args.eval_decoding in ("choice_logprob", "choice_risk"):
                            predictions = _score_numeric_options(
                                wrapper, tokenizer, input_ids, attn_mask, bv,
                                args, device=device)
                            responses = [
                                "" if prediction is None else str(prediction)
                                for prediction in predictions
                            ]
                        else:
                            gen_ids = wrapper.base_model.generate(
                                input_ids=input_ids,
                                attention_mask=attn_mask,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=(args.temperature > 0),
                                temperature=args.temperature if args.temperature > 0 else None,
                                top_p=args.top_p if args.temperature > 0 else None,
                                top_k=args.top_k if args.temperature > 0 else None,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            )
                            for row in range(len(bp)):
                                prompt_len = input_ids[row].shape[0]
                                generated_ids = gen_ids[row][prompt_len:]
                                response = tokenizer.decode(
                                    generated_ids, skip_special_tokens=True
                                ).strip()
                                responses.append(response)
                                predictions.append(extract_answer_number(
                                    response, valid_options=bv[row]))
                            if (args.invalid_prediction_fallback == "choice_logprob"
                                    and any(prediction is None
                                            for prediction in predictions)):
                                fallback_predictions = _score_numeric_options(
                                    wrapper, tokenizer, input_ids, attn_mask, bv,
                                    args, device=device)
                                for row, prediction in enumerate(predictions):
                                    if prediction is None:
                                        predictions[row] = fallback_predictions[row]
                                        responses[row] = (
                                            f"{responses[row]} [fallback="
                                            f"{fallback_predictions[row]}]"
                                        ).strip()

                        output_token_total += sum(
                            len(tokenizer.encode(
                                str(prediction), add_special_tokens=False))
                            for prediction in predictions
                            if prediction is not None
                        )

                        for j in range(len(bp)):
                            try:
                                response = responses[j]
                                prediction = predictions[j]

                                distribution_options = ordinal_core(
                                    bv[j] or (bg[j],))
                                distribution_records.append({
                                    "domain": domain_name,
                                    "person_id": bm[j]["person_id"],
                                    "question_id": qids[i + j],
                                    "target": bg[j],
                                    "prediction": prediction,
                                    "valid_options": distribution_options,
                                    "include_in_distribution": (
                                        bg[j] in distribution_options
                                        and (
                                            prediction is None
                                            or prediction in distribution_options
                                        )
                                    ),
                                })

                                if len(sample_outputs) < 20:
                                    sample_outputs.append({
                                        "question_id": qids[i + j],
                                        "raw_output": response[:500],
                                        "prediction": prediction,
                                        "ground_truth": bg[j],
                                        "features": list(features),
                                    })

                                if prediction is not None:
                                    y_true_all.append(str(bg[j]))
                                    y_pred_all.append(str(prediction))
                                    normalized_error = normalized_option_distance(
                                        prediction, bg[j], bv[j] or (bg[j],)
                                    )
                                    error_sum += normalized_error
                                    if prediction == bg[j]:
                                        correct += 1
                                    total += 1
                                else:
                                    invalid_predictions += 1
                                    total += 1
                                for metric_name, meta_key in [
                                    ("country_metrics", "country"),
                                    ("gender_metrics", "gender"),
                                    ("age_metrics", "age"),
                                    ("occupation_metrics", "occupation"),
                                    ("employment_metrics", "employment"),
                                ]:
                                    _update_metric_bucket(
                                        group_metrics[metric_name][bm[j][meta_key]],
                                        bg[j], prediction, bv[j])
                            except Exception as e:
                                invalid_predictions += 1
                                total += 1
                                for metric_name, meta_key in [
                                    ("country_metrics", "country"),
                                    ("gender_metrics", "gender"),
                                    ("age_metrics", "age"),
                                    ("occupation_metrics", "occupation"),
                                    ("employment_metrics", "employment"),
                                ]:
                                    _update_metric_bucket(
                                        group_metrics[metric_name][bm[j][meta_key]],
                                        bg[j], None, bv[j])
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"  Warning: batch error: {e}")
                        invalid_predictions += len(bp)
                        total += len(bp)

    finally:
        tokenizer.padding_side = old_pad_side

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    # Aggregate across ranks if distributed
    use_ddp = world_size > 1
    if use_ddp:
        # Reduce scalar metrics
        metrics_tensor = torch.tensor(
            [correct, total, invalid_predictions, error_sum],
            dtype=torch.float64, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        correct = int(metrics_tensor[0].item())
        total = int(metrics_tensor[1].item())
        invalid_predictions = int(metrics_tensor[2].item())
        error_sum = metrics_tensor[3].item()

        runtime_tensor = torch.tensor(
            [inference_seconds], dtype=torch.float64, device=device)
        dist.all_reduce(runtime_tensor, op=dist.ReduceOp.MAX)
        inference_seconds = runtime_tensor.item()
        token_tensor = torch.tensor(
            [input_token_total, output_token_total],
            dtype=torch.int64, device=device)
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        input_token_total = int(token_tensor[0].item())
        output_token_total = int(token_tensor[1].item())

        # Gather y_true/y_pred lists on rank 0 for F1 computation
        # Use gather_object for variable-length lists
        gathered_yt = [None] * world_size if is_main_process(rank) else None
        gathered_yp = [None] * world_size if is_main_process(rank) else None
        dist.gather_object(y_true_all, gathered_yt, dst=0)
        dist.gather_object(y_pred_all, gathered_yp, dst=0)

        if is_main_process(rank):
            y_true_all = [item for sublist in gathered_yt for item in sublist]
            y_pred_all = [item for sublist in gathered_yp for item in sublist]

        gathered_distribution = (
            [None] * world_size if is_main_process(rank) else None
        )
        dist.gather_object(
            distribution_records, gathered_distribution, dst=0
        )
        if is_main_process(rank):
            distribution_records = [
                item
                for rank_records in gathered_distribution
                for item in rank_records
            ]

        # Gather sample outputs on rank 0
        gathered_samples = [None] * world_size if is_main_process(rank) else None
        dist.gather_object(sample_outputs, gathered_samples, dst=0)
        if is_main_process(rank):
            sample_outputs = [s for sublist in gathered_samples for s in sublist][:20]

        gathered_groups = [None] * world_size if is_main_process(rank) else None
        plain_groups = {
            metric_name: dict(buckets)
            for metric_name, buckets in group_metrics.items()
        }
        dist.gather_object(plain_groups, gathered_groups, dst=0)
        if is_main_process(rank):
            merged_groups = {
                "country_metrics": defaultdict(_new_metric_bucket),
                "gender_metrics": defaultdict(_new_metric_bucket),
                "age_metrics": defaultdict(_new_metric_bucket),
                "occupation_metrics": defaultdict(_new_metric_bucket),
                "employment_metrics": defaultdict(_new_metric_bucket),
            }
            for rank_groups in gathered_groups:
                for metric_name, buckets in rank_groups.items():
                    for key, bucket in buckets.items():
                        _merge_metric_buckets(
                            merged_groups[metric_name][key], bucket)
            group_metrics = merged_groups

    acc = correct / total if total > 0 else 0
    valid_count = total - invalid_predictions
    mae = error_sum / valid_count if valid_count > 0 else 0
    valid_rate = valid_count / total if total > 0 else 0
    macro_f1, micro_f1 = _macro_micro_f1(y_true_all, y_pred_all)
    finalized_groups = (
        _finalize_group_metrics(group_metrics) if is_main_process(rank) else {}
    )
    distribution_metrics = (
        compute_distribution_metrics(distribution_records)
        if is_main_process(rank) else {}
    )

    if is_main_process(rank):
        print(f"  {desc}: Acc={acc:.4f}, MAE={mae:.4f}, "
              f"MacroF1={macro_f1:.4f}, MicroF1={micro_f1:.4f}, "
              f"ValidRate={valid_rate:.4f} "
              f"({correct}/{total}, invalid={invalid_predictions})")

    return {
        "accuracy": acc, "correct": correct, "total": total,
        "invalid_predictions": invalid_predictions,
        "valid_rate": valid_rate, "mae": mae,
        "option_distance": mae,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "valid_count": valid_count,
        "sample_outputs": sample_outputs,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "distribution_records": distribution_records,
        "error_sum": error_sum,
        "runtime_seconds": inference_seconds,
        "samples_per_second": (
            total / inference_seconds if inference_seconds > 0 else 0.0
        ),
        "token_usage": {
            "input_tokens": input_token_total,
            "output_tokens": output_token_total,
            "total_tokens": input_token_total + output_token_total,
            "tokens_per_second": (
                (input_token_total + output_token_total) / inference_seconds
                if inference_seconds > 0 else 0.0
            ),
        },
        **distribution_metrics,
        **finalized_groups,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train + Evaluate Multi-LoRA on SocioBench")

    # paths
    parser.add_argument("--sociobench_root", type=str,
                        default="data/SocioBench")
    parser.add_argument("--model_name", type=str,
                        default="models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sociobench/llama")

    # domains
    parser.add_argument("--domains", type=str, default="all",
                        help="Comma-separated domain names or 'all'")
    parser.add_argument("--dataset_size", type=int, default=500,
                        choices=[500, 5000, 50000])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--question_split_ratio", type=float, default=0.8,
                        help="Fraction of questions used for training (rest for test)")
    parser.add_argument("--max_train_respondents_per_domain", type=int,
                        default=0,
                        help="Limit sampled train respondents per domain. "
                             "0 uses all train respondents.")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=None,
        help="Linear projection attribute names that receive every task and "
             "knowledge LoRA. The default is gate_proj up_proj down_proj.")
    parser.add_argument("--feature_dimensions", type=str, nargs="+",
                        default=DEFAULT_FEATURE_DIMENSIONS)
    parser.add_argument("--lora_weight_mode", type=str, default="uniform",
                        choices=["uniform", "static", "learned_static",
                                 "residual_gate"])
    parser.add_argument("--lora_weight_normalize", type=str,
                        default="sum_to_active_count",
                        choices=["sum_to_active_count", "sum_to_one", "none"])
    parser.add_argument(
        "--lora_residual_gate_preserve_total", action="store_true",
        help="Keep the active residual-gate weights at their initial total.")
    parser.add_argument("--lora_composition_mode", type=str, default="sum",
                        choices=["sum", "orthogonal_projection",
                                 "task_knowledge_projection",
                                 "explicit_task_knowledge_projection"])
    parser.add_argument("--lora_dimension_order", type=str, nargs="+",
                        default=None)
    parser.add_argument("--lora_orthogonal_eps", type=float, default=1e-6)
    parser.add_argument("--lora_orthogonal_strength", type=float, default=1.0)
    parser.add_argument("--lora_soft_orthogonal_lambda", type=float, default=0.0)
    parser.add_argument(
        "--lora_soft_orthogonal_mode", type=str, default="a_frobenius",
        choices=["a_frobenius", "delta_cosine"],
        help="Soft task-demographic overlap penalty used by the shared trainer.")
    parser.add_argument(
        "--lora_orthogonal_grad_log_interval", type=int, default=0,
        help="Log soft-orthogonality gradient diagnostics every N steps.")
    parser.add_argument("--lora_task_scale", type=float, default=1.0)
    parser.add_argument("--lora_knowledge_scale", type=float, default=1.0)
    parser.add_argument("--task_lora_name", type=str, default="task_shared")
    parser.add_argument("--lora_train_stage", type=str, default="joint",
                        choices=["joint", "task_only", "knowledge_only"])
    parser.add_argument("--load_task_lora_dir", type=str, default=None)
    parser.add_argument("--freeze_task_lora", action="store_true")

    # training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_micro_batch_size", type=int, default=0)
    parser.add_argument("--dynamic_train_padding", action="store_true",
                        help="Pad training examples to each batch maximum.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_weight_learning_rate", type=float, default=0.0,
                        help="Optional separate LR for residual gate weights.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Learning-rate schedule applied per optimizer step.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup ratio used when warmup_steps is negative.")
    parser.add_argument("--warmup_steps", type=int, default=-1,
                        help="Explicit warmup optimizer steps. Negative uses warmup_ratio.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="Final LR as a ratio of learning_rate for decay schedules.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples_per_group", type=int, default=5000)
    parser.add_argument("--train_answer_value_only", action="store_true")
    parser.add_argument("--task_train_data_mode", type=str, default="grouped",
                        choices=["grouped", "global_random"])
    parser.add_argument("--task_user_pool", type=str, default="train",
                        choices=["train", "all"])
    parser.add_argument("--task_answer_mode", type=str, default="real",
                        choices=["real", "balanced_random"])
    parser.add_argument("--task_max_samples", type=int, default=0)
    parser.add_argument("--task_balance_answers", action="store_true")
    parser.add_argument(
        "--split_manifest", type=str, default="",
        help=("Use the exact train, held-out respondent, and question IDs "
              "from a matched-split manifest."))
    parser.add_argument(
        "--min_required_feature_dimensions", type=int, default=3,
        help=("Minimum number of available values among --feature_dimensions "
              "needed to train or evaluate a respondent."))
    parser.add_argument(
        "--data_preflight_only", action="store_true",
        help="Validate and save the data split, then exit before loading a model.")
    parser.add_argument("--trainable_feature_dimensions", type=str,
                        nargs="+", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offload_optimizer_state", action="store_true",
                        help="Move AdamW optimizer state tensors to CPU after each optimizer step.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Deprecated; checkpoint cadence is controlled by --num_checkpoints.")
    parser.add_argument("--num_checkpoints", type=int, default=10,
                        help="Save this many evenly spaced resumable checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to output_dir/checkpoints/latest for resuming training.")
    parser.add_argument("--no_auto_resume", action="store_true",
                        help="Disable automatic resume from output_dir/checkpoints/latest.")

    # eval
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_eval_respondents_per_domain", type=int, default=0,
                        help="Limit respondents per domain for evaluation. 0 uses all.")
    parser.add_argument(
        "--eval_respondent_split", type=str, default="test",
        choices=["train", "test"],
        help=("Choose respondents for evaluation. Both choices use held-out "
              "questions; 'train' is intended for parameter selection."))
    parser.add_argument(
        "--eval_include_person_ids_manifest", type=str, default="",
        help=("Restrict evaluation to each domain's eval_person_ids in a "
              "matched-split manifest."))
    parser.add_argument(
        "--eval_exclude_person_ids_manifest", type=str, default="",
        help=("Exclude each domain's eval_person_ids in a matched-split "
              "manifest. Intended for disjoint calibration."))
    parser.add_argument(
        "--eval_sample_seed", type=int, default=-1,
        help="Sampling seed for limited evaluation. Negative uses --seed.")
    parser.add_argument("--eval_exp3_only", action="store_true",
                        help="During evaluation, run only Exp3: LoRA-only on the test split.")
    parser.add_argument(
        "--eval_experiments", nargs="+", default=None,
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Optional subset of experiments to evaluate.")
    parser.add_argument(
        "--eval_lora_components", type=str, default="full",
        choices=["full", "task_only", "knowledge_only"],
        help="Select the adapters active in Exp3/Exp4 evaluation.")
    parser.add_argument(
        "--eval_knowledge_scales", type=float, nargs="+", default=None,
        help=("Evaluate Exp3 at several knowledge scales in one model load. "
              "Intended for validation-set parameter selection."))
    parser.add_argument(
        "--eval_domain_knowledge_scales_json", type=str, default="",
        help=("JSON object mapping domain names to validation-selected "
              "knowledge scales."))
    parser.add_argument(
        "--save_prediction_records", action="store_true",
        help=("Save per-person, per-question targets and predictions for "
              "reproducible offline evaluation-subset analysis."))
    parser.add_argument("--max_new_tokens", type=int, default=5000)
    parser.add_argument(
        "--eval_decoding", type=str, default="generate",
        choices=["generate", "choice_logprob", "choice_risk"],
        help=("Decode freely or constrain prediction to the highest-probability "
              "numeric option, optionally minimizing ordinal risk."))
    parser.add_argument(
        "--eval_mae_risk_weight", type=float, default=0.0,
        help=("For choice_risk decoding, weight expected normalized option "
              "distance relative to zero-one classification risk."))
    parser.add_argument(
        "--eval_choice_temperature", type=float, default=1.0,
        help="Temperature used to normalize legal-option scores into probabilities.")
    parser.add_argument(
        "--invalid_prediction_fallback", type=str, default="none",
        choices=["none", "choice_logprob"],
        help="Use legal-option scoring only when free generation is invalid.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    # mode
    parser.add_argument("--mode", type=str, default="train_eval",
                        choices=["train_eval", "train_only", "eval_only"],
                        help="train_eval: train then eval; train_only; eval_only (needs --load_lora_dir)")
    parser.add_argument("--load_lora_dir", type=str, default="",
                        help="Load pre-trained LoRA weights (for eval_only)")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.eval_domain_knowledge_scales = {}
    if args.eval_domain_knowledge_scales_json:
        with open(args.eval_domain_knowledge_scales_json) as f:
            loaded_scales = json.load(f)
        args.eval_domain_knowledge_scales = {
            str(domain): float(scale)
            for domain, scale in loaded_scales.items()
        }
    if args.lora_weight_mode == "learned_static":
        args.lora_weight_mode = "static"

    # ---- Distributed setup ----
    rank, world_size, local_rank = setup_distributed()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if is_main_process(rank):
        print("=" * 70)
        print("SocioBench Multi-LoRA Training + Evaluation")
        print(f"  Distributed: {world_size} GPU(s)")
        print("=" * 70)
        for k, v in sanitize_for_publication(vars(args)).items():
            print(f"  {k}: {v}")
        print("=" * 70)

    # Determine domains
    if args.domains == "all":
        domain_names = list(SOCIOBENCH_DOMAINS.keys())
    else:
        domain_names = [d.strip() for d in args.domains.split(",")]
        for d in domain_names:
            if d not in SOCIOBENCH_DOMAINS:
                if is_main_process(rank):
                    print(f"Error: unknown domain '{d}'")
                cleanup_distributed()
                return

    split_manifest = None
    if args.split_manifest:
        with open(args.split_manifest, encoding="utf-8") as handle:
            split_manifest = json.load(handle)
        manifest_dimensions = split_manifest.get("feature_dimensions")
        if (manifest_dimensions
                and list(manifest_dimensions) != list(args.feature_dimensions)):
            raise ValueError(
                "Split manifest feature dimensions differ from the run: "
                f"{manifest_dimensions} != {args.feature_dimensions}")

    # ---- Load and split data ----
    if is_main_process(rank):
        print("\n--- Loading and splitting SocioBench data ---")
    domain_splits = {}  # domain -> (train, test, qa_data, domain_id)
    all_train_groups = []

    for domain_name in domain_names:
        domain_info = SOCIOBENCH_DOMAINS[domain_name]
        domain_id = domain_info["domain_id"]

        try:
            qa_data = load_sociobench_qa(args.sociobench_root, domain_name)
            gt = load_sociobench_ground_truth(
                args.sociobench_root, domain_name, args.dataset_size)
        except FileNotFoundError as e:
            if is_main_process(rank):
                print(f"  Skipping {domain_name}: {e}")
            continue

        manifest_domain = (
            split_manifest.get("domains", {}).get(domain_name)
            if split_manifest else None
        )
        if split_manifest and not manifest_domain:
            raise ValueError(
                f"Split manifest has no entry for domain '{domain_name}'")
        if manifest_domain:
            respondents_by_id = {
                str(row.get("person_id")): row for row in gt
            }
            questions_by_id = {
                str(row.get("question_id", "")): row for row in qa_data
            }
            train_ids = [
                str(person_id)
                for person_id in manifest_domain.get("train_person_ids", [])
            ]
            test_ids = [
                str(person_id)
                for person_id in manifest_domain.get("held_out_person_ids", [])
            ]
            eval_ids = {
                str(person_id)
                for person_id in manifest_domain.get("eval_person_ids", [])
            }
            overlap = set(train_ids) & eval_ids
            if overlap:
                raise ValueError(
                    f"{domain_name}: split manifest leaks {len(overlap)} "
                    "evaluation respondents into training")
            missing_people = [
                person_id for person_id in train_ids + test_ids
                if person_id not in respondents_by_id
            ]
            if missing_people:
                raise ValueError(
                    f"{domain_name}: {len(missing_people)} manifest respondents "
                    "are absent from SocioBench ground truth")
            train = [respondents_by_id[person_id] for person_id in train_ids]
            test = [respondents_by_id[person_id] for person_id in test_ids]

            train_question_ids = [
                str(question_id)
                for question_id in manifest_domain.get("train_question_ids", [])
            ]
            test_question_ids = [
                str(question_id)
                for question_id in manifest_domain.get("test_question_ids", [])
            ]
            missing_questions = [
                question_id
                for question_id in train_question_ids + test_question_ids
                if question_id not in questions_by_id
            ]
            if missing_questions:
                raise ValueError(
                    f"{domain_name}: {len(missing_questions)} manifest questions "
                    "are absent from SocioBench QA data")
            qa_train = [questions_by_id[qid] for qid in train_question_ids]
            qa_test = [questions_by_id[qid] for qid in test_question_ids]
            if (args.max_train_respondents_per_domain > 0
                    and len(train) > args.max_train_respondents_per_domain):
                raise ValueError(
                    f"{domain_name}: manifest has {len(train)} training "
                    "respondents, exceeding --max_train_respondents_per_domain "
                    f"{args.max_train_respondents_per_domain}")
        else:
            qa_train, qa_test = split_questions(
                qa_data, train_ratio=args.question_split_ratio, seed=args.seed)
            train, test = split_respondents(
                gt, domain_id,
                train_ratio=args.train_ratio,
                seed=args.seed,
                feature_dimensions=args.feature_dimensions,
                min_required_feature_dimensions=(
                    args.min_required_feature_dimensions))

        if (not manifest_domain
                and args.max_train_respondents_per_domain > 0
                and len(train) > args.max_train_respondents_per_domain):
            rng = random.Random(args.seed + domain_id)
            selected_indices = set(rng.sample(
                range(len(train)), args.max_train_respondents_per_domain))
            held_out_from_train = [
                respondent for index, respondent in enumerate(train)
                if index not in selected_indices
            ]
            train = [
                respondent for index, respondent in enumerate(train)
                if index in selected_indices
            ]
            test.extend(held_out_from_train)
            rng.shuffle(test)
        if is_main_process(rank):
            print(f"  {domain_name}: {len(gt)} respondents -> "
                  f"{len(train)} train / {len(test)} test | "
                  f"{len(qa_data)} questions -> {len(qa_train)} train / {len(qa_test)} test")

        domain_splits[domain_name] = (train, test, qa_train, qa_test, domain_id)

        # Build training groups for this domain (train respondents + train questions only)
        if args.mode != "eval_only":
            groups = build_sociobench_training_groups(
                train, qa_train, domain_id, domain_name,
                feature_dimensions=args.feature_dimensions,
                min_required_feature_dimensions=(
                    args.min_required_feature_dimensions))
            all_train_groups.append(groups)

    if not domain_splits:
        cleanup_distributed()
        raise RuntimeError(
            "No SocioBench domains were loaded; check --sociobench_root: "
            f"{public_path(args.sociobench_root)}"
        )

    # Save split info (rank 0 only)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        split_info = {}
        for dn, (tr, te, qa_tr, qa_te, did) in domain_splits.items():
            split_info[dn] = {
                "domain_id": did,
                "train_count": len(tr),
                "test_count": len(te),
                "train_question_count": len(qa_tr),
                "test_question_count": len(qa_te),
                "train_person_ids": [r.get("person_id") for r in tr],
                "test_person_ids": [r.get("person_id") for r in te],
            }
        with open(os.path.join(args.output_dir, "data_split.json"), "w") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print("\nSplit info saved -> "
              f"{public_path(os.path.join(args.output_dir, 'data_split.json'))}")

    if args.data_preflight_only:
        if is_main_process(rank):
            print("Data preflight complete; model loading skipped.")
        cleanup_distributed()
        return

    # ---- Load model ----
    if is_main_process(rank):
        print(f"\nLoading model: {public_path(args.model_name)} "
              "(this may take 1-2 minutes)...")
    t0 = time.time()
    config, is_local = load_backbone_config(args.model_name)
    tokenizer = load_backbone_tokenizer(
        args.model_name, config=config, is_local=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": f"cuda:{local_rank}"} if world_size > 1 else "auto"
    model = load_backbone_model(
        args.model_name,
        config=config,
        is_local=is_local,
        device_map=device_map,
    )

    if is_main_process(rank):
        print(f"Model loaded in {time.time() - t0:.1f}s")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_names: Set[str] = set()
    for train, test, qa_train, qa_test, domain_id in domain_splits.values():
        for respondent in train + test:
            lora_names.update(get_sociobench_user_features(
                respondent.get("attributes", {}), domain_id,
                args.feature_dimensions))
    if (args.lora_composition_mode == "explicit_task_knowledge_projection"
            or args.lora_train_stage in ("task_only", "knowledge_only")
            or args.load_task_lora_dir):
        lora_names.add(args.task_lora_name)
    lora_names = set(name for name in lora_names if name)
    lora_name_list = sorted(lora_names)
    wrapper_lora_name_list = lora_name_list
    if args.lora_train_stage == "task_only":
        # Stage I never reads demographic adapters. Constructing them here can
        # consume most of an 80 GiB GPU while producing identical task updates.
        wrapper_lora_name_list = [args.task_lora_name]

    if is_main_process(rank):
        print(f"Feature dimensions ({len(args.feature_dimensions)}): "
              f"{args.feature_dimensions}")
        print(f"Features ({len(lora_name_list)}): {lora_name_list}")
        if wrapper_lora_name_list != lora_name_list:
            print("Stage-I instantiated adapters: "
                  f"{wrapper_lora_name_list}")
    wrapper = MultiLoRAModelWrapper(
        model, wrapper_lora_name_list,
        rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout, model_name=args.model_name,
        weight_mode=args.lora_weight_mode,
        weight_normalize=args.lora_weight_normalize,
        composition_mode=args.lora_composition_mode,
        dimension_order=args.lora_dimension_order or args.feature_dimensions,
        orthogonal_eps=args.lora_orthogonal_eps,
        orthogonal_strength=args.lora_orthogonal_strength,
        task_scale=args.lora_task_scale,
        knowledge_scale=args.lora_knowledge_scale,
        task_lora_name=args.task_lora_name,
        target_modules=args.lora_target_modules,
    )

    if (args.mode in ("train_eval", "train_only")
            and args.load_lora_dir
            and not args.resume_from_checkpoint):
        if is_main_process(rank):
            print("[init] loading LoRA adapters before training: "
                  f"{public_path(args.load_lora_dir)}")
        wrapper.load_all_loras(args.load_lora_dir)

    if (args.mode in ("train_eval", "train_only", "eval_only")
            and args.load_task_lora_dir
            and not args.resume_from_checkpoint):
        _load_task_lora(wrapper, args.load_task_lora_dir, args.task_lora_name)

    if (args.mode in ("train_eval", "train_only")
            and args.trainable_feature_dimensions):
        wrapper.set_trainable_feature_dimensions(
            args.trainable_feature_dimensions)

    if args.mode in ("train_eval", "train_only"):
        if args.lora_train_stage == "task_only":
            wrapper.set_trainable_lora_names(
                [args.task_lora_name], train_weighting=False)
        elif args.lora_train_stage == "knowledge_only":
            trainable_names = [
                name for name in lora_name_list
                if name != args.task_lora_name
            ]
            wrapper.set_trainable_lora_names(
                trainable_names, train_weighting=True)
        elif args.freeze_task_lora and args.task_lora_name in lora_name_list:
            trainable_names = [
                name for name in lora_name_list
                if name != args.task_lora_name
            ]
            wrapper.set_trainable_lora_names(
                trainable_names, train_weighting=True)

    # ---- Train ----
    if args.mode in ("train_eval", "train_only"):
        merged = merge_groups(all_train_groups)
        if args.lora_train_stage == "task_only":
            merged = collapse_groups_to_task_pool(merged, args)

        # Cap samples per group
        if args.max_samples_per_group > 0:
            for key in merged:
                if len(merged[key]) > args.max_samples_per_group:
                    merged[key] = random.sample(
                        merged[key], args.max_samples_per_group)

        total_samples = sum(len(v) for v in merged.values())
        if is_main_process(rank):
            print(f"\nMerged training data: {total_samples:,} QA pairs "
                  f"in {len(merged)} feature groups")
            for feat, items in sorted(merged.items(), key=lambda x: -len(x[1])):
                print(f"  {'_'.join(sorted(feat))}: {len(items)}")

        if not args.resume_from_checkpoint and not args.no_auto_resume:
            latest_ckpt = os.path.join(args.output_dir, "checkpoints", "latest")
            if os.path.isdir(latest_ckpt):
                args.resume_from_checkpoint = latest_ckpt
                if is_main_process(rank):
                    print("[checkpoint] auto-resume enabled: "
                          f"{public_path(latest_ckpt)}")

        train_start_time = time.time()
        current_train_multi_lora(
            wrapper, merged, tokenizer, args,
            use_augmented=True, use_tool_call=False)
        training_seconds = time.time() - train_start_time

    # ---- Load pre-trained (eval_only) ----
    if args.mode == "eval_only":
        requested_experiments = set(args.eval_experiments or ())
        adapter_free_eval = (
            requested_experiments
            and requested_experiments.issubset({"exp1", "exp2"})
            and not args.eval_exp3_only
            and not args.eval_knowledge_scales
        )
        if not args.load_lora_dir and not adapter_free_eval:
            if is_main_process(rank):
                print("Error: --load_lora_dir required for eval_only mode")
            cleanup_distributed()
            return
        if args.load_lora_dir:
            if is_main_process(rank):
                print(f"\nLoading LoRA from: "
                      f"{public_path(args.load_lora_dir)}")
            wrapper.load_all_loras(args.load_lora_dir)
        elif is_main_process(rank):
            print("\nAdapter-free eval_only: evaluating Exp1/Exp2")

    # ---- Evaluate ----
    if args.mode in ("train_eval", "eval_only"):
        if (args.eval_include_person_ids_manifest
                and args.eval_exclude_person_ids_manifest):
            raise ValueError(
                "Only one of eval_include_person_ids_manifest and "
                "eval_exclude_person_ids_manifest may be set")

        include_eval_ids = {}
        exclude_eval_ids = {}
        for manifest_path, destination in [
                (args.eval_include_person_ids_manifest, include_eval_ids),
                (args.eval_exclude_person_ids_manifest, exclude_eval_ids)]:
            if not manifest_path:
                continue
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            for domain_name, split in manifest.get("domains", {}).items():
                destination[domain_name] = [
                    str(person_id) for person_id in
                    split.get("eval_person_ids", [])
                ]

        if is_main_process(rank):
            print(f"\n--- Evaluating on {args.eval_respondent_split} "
                  "respondents with held-out questions ---")
        eval_start_time = time.time()
        all_results = {}

        for domain_name in domain_splits:
            train, test, qa_train, qa_test, domain_id = domain_splits[domain_name]
            eval_respondents = train if args.eval_respondent_split == "train" else test
            if domain_name in include_eval_ids:
                respondents_by_id = {
                    str(row.get("person_id")): row for row in eval_respondents
                }
                eval_respondents = [
                    respondents_by_id[person_id]
                    for person_id in include_eval_ids[domain_name]
                    if person_id in respondents_by_id
                ]
            elif domain_name in exclude_eval_ids:
                excluded = set(exclude_eval_ids[domain_name])
                eval_respondents = [
                    row for row in eval_respondents
                    if str(row.get("person_id")) not in excluded
                ]
            if (args.max_eval_respondents_per_domain > 0
                    and len(eval_respondents) > args.max_eval_respondents_per_domain):
                sample_seed = (
                    args.eval_sample_seed
                    if args.eval_sample_seed >= 0 else args.seed
                )
                rng = random.Random(sample_seed)
                eval_respondents = rng.sample(
                    eval_respondents, args.max_eval_respondents_per_domain)

            if is_main_process(rank):
                print(f"\n{'=' * 60}")
                print(f"Domain: {domain_name} "
                      f"({args.eval_respondent_split}: "
                      f"{len(eval_respondents)} respondents, "
                      f"{len(qa_test)} held-out questions)")
                print(f"{'=' * 60}")

            domain_results = {}

            experiments = [
                ("exp1_baseline", False, False, None),
                ("exp2_prompt_only", False, True, None),
                ("exp3_lora_only", True, False, args.lora_knowledge_scale),
                ("exp4_prompt_and_lora", True, True, args.lora_knowledge_scale),
            ]
            if args.eval_experiments:
                requested = {
                    "exp1": "exp1_baseline",
                    "exp2": "exp2_prompt_only",
                    "exp3": "exp3_lora_only",
                    "exp4": "exp4_prompt_and_lora",
                }
                requested_names = {
                    requested[name] for name in args.eval_experiments
                }
                experiments = [
                    experiment for experiment in experiments
                    if experiment[0] in requested_names
                ]
            if args.eval_knowledge_scales:
                experiments = []
                for scale in args.eval_knowledge_scales:
                    scale_label = f"{scale:g}".replace("-", "m").replace(".", "p")
                    experiments.append((
                        f"exp3_knowledge_scale_{scale_label}",
                        True, False, scale))
            elif args.eval_exp3_only:
                selected_scale = args.eval_domain_knowledge_scales.get(
                    domain_name, args.lora_knowledge_scale)
                experiments = [(
                    "exp3_lora_only", True, False,
                    selected_scale)]

            for exp_name, use_lora, use_prompt, knowledge_scale in experiments:
                if knowledge_scale is not None:
                    wrapper.set_composition_scales(
                        knowledge_scale=knowledge_scale)
                res = evaluate_sociobench_domain(
                    wrapper, qa_test, eval_respondents, domain_name, domain_id,
                    tokenizer, args,
                    desc=f"[{domain_name}] {exp_name}",
                    use_lora=use_lora, use_persona_prompt=use_prompt,
                    rank=rank, world_size=world_size,
                    local_rank=local_rank)
                domain_results[exp_name] = res

            all_results[domain_name] = domain_results

        # Save results (rank 0 only)
        if is_main_process(rank):
            save_results = {}
            save_samples = {}
            internal_keys = {
                "sample_outputs", "y_true_all", "y_pred_all",
                "distribution_records", "error_sum",
            }
            for domain, exps in all_results.items():
                save_results[domain] = {}
                save_samples[domain] = {}
                for exp_name, res in exps.items():
                    save_results[domain][exp_name] = {
                        k: v for k, v in res.items() if k not in internal_keys
                    }
                    save_samples[domain][exp_name] = res.get("sample_outputs", [])

            # Compute overall metrics across all domains
            exp_keys = list(next(iter(all_results.values())).keys())
            overall = {}
            prediction_records_to_save = []
            for ek in exp_keys:
                total_correct = 0
                total_count = 0
                total_invalid = 0
                total_valid = 0
                total_error_sum = 0.0
                total_runtime_seconds = 0.0
                total_input_tokens = 0
                total_output_tokens = 0
                y_true_all = []
                y_pred_all = []
                distribution_records = []
                for domain, exps in all_results.items():
                    res = exps.get(ek, {})
                    total_correct += res.get("correct", 0)
                    total_count += res.get("total", 0)
                    total_invalid += res.get("invalid_predictions", 0)
                    total_valid += res.get("valid_count", 0)
                    total_error_sum += res.get("error_sum", 0.0)
                    total_runtime_seconds += res.get("runtime_seconds", 0.0)
                    token_usage = res.get("token_usage", {})
                    total_input_tokens += token_usage.get("input_tokens", 0)
                    total_output_tokens += token_usage.get("output_tokens", 0)
                    y_true_all.extend(res.get("y_true_all", []))
                    y_pred_all.extend(res.get("y_pred_all", []))
                    experiment_records = res.get("distribution_records", [])
                    distribution_records.extend(experiment_records)
                    prediction_records_to_save.extend({
                        "experiment": ek,
                        **record,
                    } for record in experiment_records)

                acc = total_correct / total_count if total_count > 0 else 0
                mae = total_error_sum / total_valid if total_valid > 0 else 0
                valid_rate = total_valid / total_count if total_count > 0 else 0
                macro_f1, micro_f1 = _macro_micro_f1(y_true_all, y_pred_all)
                distribution_metrics = compute_distribution_metrics(
                    distribution_records)

                overall[ek] = {
                    "accuracy": acc,
                    "correct": total_correct,
                    "total": total_count,
                    "invalid_predictions": total_invalid,
                    "valid_rate": valid_rate,
                    "mae": mae,
                    "option_distance": mae,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                    "valid_count": total_valid,
                    "runtime_seconds": total_runtime_seconds,
                    "samples_per_second": (
                        total_count / total_runtime_seconds
                        if total_runtime_seconds > 0 else 0.0
                    ),
                    "token_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": (
                            total_input_tokens + total_output_tokens
                        ),
                        "tokens_per_second": (
                            (total_input_tokens + total_output_tokens)
                            / total_runtime_seconds
                            if total_runtime_seconds > 0 else 0.0
                        ),
                    },
                    **distribution_metrics,
                }

            save_results["overall"] = overall

            results_path = os.path.join(args.output_dir, "eval_results.json")
            with open(results_path, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"\nResults saved -> {public_path(results_path)}")

            samples_path = os.path.join(args.output_dir, "sample_outputs.json")
            with open(samples_path, "w") as f:
                json.dump(save_samples, f, indent=2, ensure_ascii=False)
            print(f"Sample outputs saved -> {public_path(samples_path)}")

            if args.save_prediction_records:
                records_path = os.path.join(
                    args.output_dir, "prediction_records.jsonl")
                with open(records_path, "w", encoding="utf-8") as handle:
                    for record in prediction_records_to_save:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("Prediction records saved -> "
                      f"{public_path(records_path)}")

            timing_payload = {
                "training_seconds": locals().get("training_seconds", 0.0),
                "evaluation_seconds": time.time() - eval_start_time,
                "hardware": _runtime_hardware_metadata(),
                "timing_scope": (
                    "model-ready through final prediction; model and adapter "
                    "loading excluded"
                ),
            }
            timing_path = os.path.join(args.output_dir, "run_timing.json")
            with open(timing_path, "w") as f:
                json.dump(timing_payload, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"Timing saved -> {public_path(timing_path)}")

            # Print summary table
            print("\n" + "=" * 110)
            print("EVALUATION SUMMARY (test split)")
            print("=" * 110)

            exp_short = ["Baseline", "Prompt", "LoRA", "P+LoRA"]
            ovr = save_results["overall"]

            def _print_metric_table(metric_name, metric_key):
                print(f"\n--- {metric_name} ---")
                print(f"{'Domain':<20} " + "  ".join(f"{s:>10}" for s in exp_short))
                print("-" * 70)
                for domain, exps in all_results.items():
                    vals = []
                    for ek in exp_keys:
                        res = exps.get(ek, {})
                        vals.append(f"{res.get(metric_key, 0):.4f}")
                    print(f"{domain:<20} " + "  ".join(f"{v:>10}" for v in vals))
                print("-" * 70)
                vals = [f"{ovr[ek][metric_key]:.4f}" for ek in exp_keys]
                print(f"{'OVERALL':<20} " + "  ".join(f"{v:>10}" for v in vals))

            _print_metric_table("Accuracy", "accuracy")
            _print_metric_table("MAE", "mae")
            _print_metric_table("Option Distance", "option_distance")
            _print_metric_table("Macro F1", "macro_f1")
            _print_metric_table("Micro F1", "micro_f1")
            _print_metric_table("Valid Rate", "valid_rate")

            print("=" * 110)

    if is_main_process(rank):
        print("\nDone!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
