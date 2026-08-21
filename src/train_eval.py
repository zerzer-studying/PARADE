"""WVS training and evaluation entry point for PARADE.

The main scripts use complete profiles over eight dimensions: gender, age
group, country, religion, education, marital status, employment status, and
urban/rural residence.  Training is split into a demographic-agnostic Task
LoRA stage, using balanced random legal option identifiers in the main runs,
and a Demographic LoRA stage.  In the second stage the Task LoRA is frozen,
the eight attribute-value adapters selected by a respondent are co-trained,
and dimension-level weights plus soft task/demographic orthogonality
calibrate their composition.

Evaluation uses the same question-only prompt for parameter-based methods and
the respondent-specific composition described above.  See ``lora.py`` for
composition and PEFT export details.
"""

import os
import shutil
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
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import random
from functools import partial


def _ddp_env():
    """Read torchrun-injected env vars. Returns (rank, world_size, local_rank, is_ddp)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world_size, local_rank, world_size > 1
    return 0, 1, 0, False


RANK, WORLD_SIZE, LOCAL_RANK, IS_DDP = _ddp_env()
IS_MAIN = RANK == 0

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
from lora import MultiLoRAModelWrapper
from dataloader import QADataset
from evaluate import evaluate, evaluate_choice_logprob, evaluate_with_persona_prompt
from config import *
from utils import *
from model_loading import (
    load_backbone_config,
    load_backbone_model,
    load_backbone_tokenizer,
)
from anonymization import public_path, sanitize_for_publication

def _collate(batch, *, pad_token_id: int = 0, pad_to_multiple: int = 8):
    max_length = max(ex['input_ids'].numel() for ex in batch)
    if pad_to_multiple > 1:
        max_length = int(
            math.ceil(max_length / pad_to_multiple) * pad_to_multiple
        )
    result = {
        'input_ids': torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long
        ),
        'labels': torch.full(
            (len(batch), max_length), -100, dtype=torch.long
        ),
        'attention_mask': torch.zeros(
            (len(batch), max_length), dtype=torch.long
        ),
    }
    for row, example in enumerate(batch):
        length = example['input_ids'].numel()
        for key in result:
            result[key][row, :length] = example[key]
    return result


def _batch_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


def _slice_batch(batch, start: int, end: int):
    return {
        k: v[start:end]
        for k, v in batch.items()
    }


def _move_optimizer_state(optimizer, device, *, only_with_grad: bool = False):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if only_with_grad and p.grad is None:
                continue
            state = optimizer.state.get(p)
            if not state:
                continue
            for k, v in list(state.items()):
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device, non_blocking=True)


def _move_optimizer_state_for_params(optimizer, params, device):
    for p in params:
        state = optimizer.state.get(p)
        if not state:
            continue
        for k, v in list(state.items()):
            if torch.is_tensor(v) and v.device != device:
                state[k] = v.to(device, non_blocking=True)


def _offload_optimizer_state_to_cpu(optimizer):
    _move_optimizer_state(optimizer, torch.device("cpu"))


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _loss_function_description():
    return (
        "AutoModelForCausalLM causal language modeling loss from model output "
        "`out.loss`; labels use -100 masking from QADataset."
    )


def _json_dump_atomic(payload, path):
    # Multiple launchers can briefly overlap on shared storage during retries.
    # Keep each writer's temporary file private before the atomic replacement.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(sanitize_for_publication(payload), f,
                      indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _save_training_run_config(args):
    if not IS_MAIN:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "loss_function": _loss_function_description(),
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "optimizer": "AdamW",
        "optimizer_weight_decay": 0.01,
        "lora_weight_mode": args.lora_weight_mode,
        "lora_weight_normalize": args.lora_weight_normalize,
        "lora_composition_mode": args.lora_composition_mode,
        "lora_target_modules": args.lora_target_modules,
        "lora_dimension_order": args.lora_dimension_order,
        "lora_orthogonal_eps": args.lora_orthogonal_eps,
        "lora_orthogonal_strength": args.lora_orthogonal_strength,
        "lora_soft_orthogonal_lambda": args.lora_soft_orthogonal_lambda,
        "lora_task_scale": args.lora_task_scale,
        "lora_knowledge_scale": args.lora_knowledge_scale,
        "task_lora_name": args.task_lora_name,
        "lora_train_stage": args.lora_train_stage,
        "load_task_lora_dir": args.load_task_lora_dir,
        "freeze_task_lora": args.freeze_task_lora,
        "task_train_data_mode": args.task_train_data_mode,
        "task_user_pool": args.task_user_pool,
        "task_answer_mode": args.task_answer_mode,
        "task_max_samples": args.task_max_samples,
        "task_balance_answers": args.task_balance_answers,
        "dynamic_train_padding": args.dynamic_train_padding,
        "trainable_feature_dimensions": args.trainable_feature_dimensions,
        "train_answer_value_only": args.train_answer_value_only,
        "train_micro_batch_size": args.train_micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps": args.save_steps,
        "num_checkpoints": args.num_checkpoints,
        "args": vars(args),
    }
    _json_dump_atomic(payload, os.path.join(args.output_dir, "training_config.json"))


def _write_realtime_training_state(args, latest_metrics):
    if not IS_MAIN:
        return
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    with open(metrics_path, "a") as f:
        f.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())

    live_state = {
        "loss_function": _loss_function_description(),
        "learning_rate": latest_metrics.get("learning_rate", args.learning_rate),
        "latest_metrics": latest_metrics,
        "metrics_file": metrics_path,
    }
    _json_dump_atomic(live_state, os.path.join(args.output_dir, "training_state_live.json"))


def _save_training_checkpoint(wrapper, optimizer, args, *, epoch: int,
                              epoch_step: int, global_step: int,
                              latest_metrics: Optional[dict] = None,
                              scheduler=None,
                              optimizer_step_index: int = 0,
                              checkpoint_name: str = "latest"):
    if not IS_MAIN:
        return
    ckpt_root = os.path.join(args.output_dir, "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, checkpoint_name)
    tmp_dir = os.path.join(ckpt_root, f"{checkpoint_name}.tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    lora_dir = os.path.join(tmp_dir, "lora")
    wrapper.save_all_loras(lora_dir)
    state = {
        "epoch": epoch,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "loss_function": _loss_function_description(),
        "learning_rate": (
            latest_metrics.get("learning_rate", args.learning_rate)
            if latest_metrics else args.learning_rate
        ),
        "latest_metrics": latest_metrics,
        "optimizer": _to_cpu(optimizer.state_dict()),
        "optimizer_step_index": optimizer_step_index,
    }
    if scheduler is not None:
        state["scheduler"] = _to_cpu(scheduler.state_dict())
    torch.save(state, os.path.join(tmp_dir, "training_state.pt"))
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)
    print(f"[checkpoint] saved -> {public_path(ckpt_dir)}")

    if checkpoint_name != "latest":
        latest_dir = os.path.join(ckpt_root, "latest")
        latest_tmp_dir = os.path.join(ckpt_root, "latest.tmp")
        if os.path.exists(latest_tmp_dir):
            shutil.rmtree(latest_tmp_dir)
        shutil.copytree(ckpt_dir, latest_tmp_dir)
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        os.replace(latest_tmp_dir, latest_dir)
        print(f"[checkpoint] updated -> {public_path(latest_dir)}")


def _load_training_checkpoint(wrapper, optimizer, checkpoint_dir: str,
                              scheduler=None):
    lora_dir = os.path.join(checkpoint_dir, "lora")
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {lora_dir}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Training state not found: {state_path}")
    wrapper.load_all_loras(lora_dir)
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    optimizer_step_index = state.get("optimizer_step_index")
    if optimizer_step_index is None:
        optimizer_step_index = state.get("scheduler", {}).get("last_epoch", 0)
    return (
        int(state.get("epoch", 0)),
        int(state.get("epoch_step", 0)),
        int(state.get("global_step", 0)),
        int(optimizer_step_index),
    )


def _build_lr_scheduler(optimizer, args, total_batches: int):
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(total_batches / grad_accum_steps))
    total_steps = max(1, steps_per_epoch * max(1, args.num_epochs))
    warmup_steps = args.warmup_steps
    if warmup_steps < 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = min(max(0, warmup_steps), total_steps)
    min_lr_ratio = max(0.0, min(1.0, args.min_lr_ratio))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if args.lr_scheduler_type == "constant":
            return 1.0
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        if args.lr_scheduler_type == "linear":
            factor = 1.0 - progress
        elif args.lr_scheduler_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown lr_scheduler_type: {args.lr_scheduler_type}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, total_steps, warmup_steps


def _build_checkpoint_steps(total_steps: int, num_checkpoints: int):
    num_checkpoints = max(0, int(num_checkpoints))
    if num_checkpoints == 0:
        return []
    count = min(num_checkpoints, max(1, total_steps))
    return sorted({
        min(total_steps, max(1, math.ceil(total_steps * i / count)))
        for i in range(1, count + 1)
    })


def _build_group_batch_schedule(loaders, *, seed: int, epoch: int):
    """Return a shuffled epoch schedule that exhausts every group loader once."""
    schedule = []
    ordered_keys = sorted(
        loaders, key=lambda feature_key: tuple(sorted(map(str, feature_key))))
    for feature_key in ordered_keys:
        schedule.extend([feature_key] * len(loaders[feature_key]))
    random.Random(seed + epoch).shuffle(schedule)
    return schedule


def _advance_loader_for_resume(schedule, loader_iters, *, steps: int):
    if steps < 0 or steps > len(schedule):
        raise ValueError(
            f"Invalid resume step {steps}; epoch has {len(schedule)} batches")
    for feature_key in schedule[:steps]:
        try:
            next(loader_iters[feature_key])
        except StopIteration as exc:
            raise RuntimeError(
                "Group loader was exhausted while reconstructing checkpoint "
                f"state for {sorted(feature_key)}") from exc


def _collapse_groups_to_joint_adapters(groups: dict) -> dict:
    """Train each full feature combination as a single joint adapter.

    Example:
      frozenset({"female", "young", "asia"})
      -> frozenset({"asia__female__young"})

    This is the reference model needed for interaction residual analysis:
    Delta_joint is compared against the additive Delta_female + Delta_young
    + Delta_asia approximation outside the training loop.
    """
    collapsed = defaultdict(list)
    for features, items in groups.items():
        collapsed[frozenset({joint_adapter_name(features)})].extend(items)
    return dict(collapsed)


def _sample_train_user_ids(train_uids, args):
    train_uids = list(train_uids)
    max_train_users = int(getattr(args, "max_train_users", 0) or 0)
    if max_train_users <= 0 or max_train_users >= len(train_uids):
        return train_uids
    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(train_uids, max_train_users))
    if IS_MAIN:
        print(f"Sampled train users: {len(sampled):,} / {len(train_uids):,} "
                  f"(seed={args.seed})")
    return sampled


def _should_apply_group_sample_cap(args) -> bool:
    """Return whether max_samples_per_group applies to this train stage."""
    is_global_task = (
        args.lora_train_stage == "task_only"
        and args.task_train_data_mode == "global_random"
    )
    return args.max_samples_per_group > 0 and not is_global_task


def _feature_dimension_name(feature: str) -> Optional[str]:
    if feature in ("male", "female"):
        return "gender"
    if feature in ("young", "middle_aged", "old"):
        return "age"
    if feature in ("age_young", "age_middle", "age_senior"):
        return "age_group"
    if feature in ("asia", "europe", "north_america", "south_america",
                   "africa", "oceania"):
        return "region"
    if feature.startswith("country_"):
        return "country"
    if feature.startswith("edu_"):
        return "education"
    if feature.startswith("rel_"):
        return "religion"
    if feature.startswith("marital_"):
        return "marital_status"
    if feature.startswith("employment_"):
        return "employment"
    if feature.startswith("occupation_"):
        return "occupation"
    if feature in ("urban", "rural"):
        return "urban_rural"
    return None


def _present_feature_dimensions(features, requested_dimensions) -> Set[str]:
    requested = set(requested_dimensions or [])
    present = set()
    for feature in features:
        dim = _feature_dimension_name(feature)
        if dim == "marital_status" and "marital" in requested:
            present.add("marital")
        if dim in requested:
            present.add(dim)
    return present


def _filter_complete_feature_users(wvs_data, train_uids, args):
    dimensions = list(getattr(args, "feature_dimensions", None) or [])
    if (not getattr(args, "require_complete_feature_dimensions", False)
            or not dimensions):
        return list(train_uids)

    required = set(dimensions)
    complete_uids = []
    missing_counts = Counter()

    for uid in train_uids:
        if uid >= len(wvs_data):
            missing_counts.update(required)
            continue
        features = get_user_features(
            wvs_data[uid],
            dimensions=dimensions,
            raw_education=args.raw_education_features,
        )
        present = _present_feature_dimensions(features, dimensions)
        missing = required - present
        if missing:
            missing_counts.update(missing)
        else:
            complete_uids.append(uid)

    if not complete_uids:
        raise ValueError(
            "No train users have complete requested feature dimensions: "
            f"{dimensions}"
        )

    if IS_MAIN:
        total = len(train_uids)
        print(
            "Complete feature-dimension train users: "
            f"{len(complete_uids):,} / {total:,} "
            f"(required dims={dimensions})"
        )
        if missing_counts:
            print("Missing train users by dimension:")
            for dim in dimensions:
                print(f"  {dim}: {missing_counts.get(dim, 0):,}")

    return complete_uids


def build_global_task_qa(wvs_data: list, user_indices, question_ids: list,
                         nature_options: dict, args) -> dict:
    """Build one global task-only pool, detached from demographic groups.

    The balanced_random answer mode uses WVS questions/options but replaces
    respondent answers with uniformly sampled valid option labels. That keeps
    task_shared focused on the multiple-choice output protocol instead of
    fitting population-specific WVS preferences.
    """
    rng = random.Random(args.seed)
    items = []
    user_set = set(user_indices)
    answer_mode = getattr(args, "task_answer_mode", "real")

    for idx in tqdm(sorted(user_set), desc="Building global task QA"):
        if idx >= len(wvs_data):
            continue
        row = wvs_data[idx]
        for qid in question_ids:
            if qid not in nature_options:
                continue
            q_info = nature_options[qid]
            opts = q_info.get("options", {})
            valid_keys = [
                str(k) for k in sorted(
                    opts.keys(),
                    key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0,
                )
                if str(k).lstrip("-").isdigit() and int(k) >= 0
            ]
            if not valid_keys:
                continue

            if answer_mode == "balanced_random":
                answer_value = rng.choice(valid_keys)
            else:
                val = row.get(qid, "")
                if not val:
                    continue
                try:
                    v = int(float(val))
                except (ValueError, TypeError):
                    continue
                if v < 0:
                    continue
                answer_value = str(v)
                if answer_value not in opts:
                    continue

            answer_text = opts.get(answer_value)
            if not answer_text:
                continue

            q_text = q_info["question_text"]
            opts_fmt = "\n".join(
                f"{k}. {t}" for k, t in
                sorted(opts.items(),
                       key=lambda x: int(x[0]) if x[0].lstrip("-").isdigit() else 0)
            )
            items.append({
                "question_id": qid,
                "question": f"{q_text}\nOptions:\n{opts_fmt}",
                "answer": answer_text,
                "answer_value": answer_value,
            })

    if getattr(args, "task_balance_answers", False):
        by_answer = defaultdict(list)
        for item in items:
            by_answer[item["answer_value"]].append(item)
        for bucket in by_answer.values():
            rng.shuffle(bucket)
        if args.task_max_samples > 0:
            per_answer = max(1, args.task_max_samples // max(len(by_answer), 1))
            balanced = []
            leftovers = []
            for bucket in by_answer.values():
                balanced.extend(bucket[:per_answer])
                leftovers.extend(bucket[per_answer:])
            rng.shuffle(leftovers)
            balanced.extend(leftovers[:max(0, args.task_max_samples - len(balanced))])
            items = balanced[:args.task_max_samples]
        elif by_answer:
            per_answer = min(len(bucket) for bucket in by_answer.values())
            items = [
                item
                for bucket in by_answer.values()
                for item in bucket[:per_answer]
            ]
    elif args.task_max_samples > 0 and len(items) > args.task_max_samples:
        items = rng.sample(items, args.task_max_samples)

    rng.shuffle(items)
    print(f"  {len(items):,} global task QA pairs "
          f"(answer_mode={answer_mode}, balance={args.task_balance_answers})")
    return {frozenset({args.task_lora_name}): items}


def _training_active_loras(feat_key, args):
    if getattr(args, "lora_train_stage", "joint") == "task_only":
        return frozenset({args.task_lora_name})
    return feat_key


def _configure_stage_trainability(wrapper, args, lora_names,
                                  knowledge_trainable_names=None):
    """Apply the trainable-parameter contract for the selected LoRA stage."""
    if args.lora_train_stage == "task_only":
        wrapper.set_trainable_lora_names(
            [args.task_lora_name], train_weighting=False)
    elif args.lora_train_stage == "knowledge_only":
        trainable_names = knowledge_trainable_names
        if trainable_names is None:
            trainable_names = [
                name for name in lora_names
                if name != args.task_lora_name
            ]
        wrapper.set_trainable_lora_names(
            trainable_names, train_weighting=True)
    elif args.lora_train_stage == "weighting_only":
        wrapper.set_trainable_lora_names([], train_weighting=True)
    elif args.freeze_task_lora and args.task_lora_name in lora_names:
        trainable_names = [
            name for name in lora_names
            if name != args.task_lora_name
        ]
        wrapper.set_trainable_lora_names(
            trainable_names, train_weighting=True)


def _eval_lora_component_mode(args) -> str:
    return getattr(args, "eval_lora_components", "full")


def _eval_merge_dir(args) -> str:
    mode = _eval_lora_component_mode(args)
    suffix = "" if mode == "full" else f"_{mode}"
    return os.path.join(args.output_dir, f"merged_combos{suffix}")


def _eval_extra_active_loras(args) -> Optional[List[str]]:
    mode = _eval_lora_component_mode(args)
    if mode in ("full", "task_only"):
        if args.lora_composition_mode == "explicit_task_knowledge_projection":
            return [args.task_lora_name]
    return None


def _eval_extra_lora_count(args) -> int:
    mode = _eval_lora_component_mode(args)
    if mode in ("full", "task_only"):
        return 1 if args.lora_composition_mode == "explicit_task_knowledge_projection" else 0
    return 0


def _collect_feature_names_for_users(wvs_data, user_indices, args) -> Set[str]:
    names: Set[str] = set()
    for idx in user_indices:
        if idx >= len(wvs_data):
            continue
        features = get_user_features(
            wvs_data[idx],
            dimensions=args.feature_dimensions,
            raw_education=args.raw_education_features,
        )
        names.update(features)
    return names


def _load_task_lora(wrapper, task_lora_dir: str, task_lora_name: str):
    if not task_lora_dir:
        return
    direct = os.path.join(task_lora_dir, "adapter_config.json")
    if os.path.exists(direct):
        load_dir = task_lora_dir
    else:
        load_dir = os.path.join(task_lora_dir, task_lora_name)
    print(f"[task] loading task LoRA '{task_lora_name}' from "
          f"{public_path(load_dir)}")
    wrapper.load_lora(load_dir, task_lora_name)

# ======================================================================
# Training
# ======================================================================

def train_multi_lora(wrapper: MultiLoRAModelWrapper, groups: dict,
                     tokenizer, args, use_augmented: bool = True,
                     use_tool_call: bool = False):
    device = wrapper.base_model.get_input_embeddings().weight.device
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    collate_fn = partial(_collate, pad_token_id=pad_token_id)
    pin_memory = torch.cuda.is_available()

    if IS_MAIN:
        print("\nTokenizing training data …")
        if args.train_answer_value_only:
            fmt = "Answer value only"
        elif use_tool_call:
            fmt = "Tool-call (Qwen native format)"
        elif use_augmented:
            fmt = "Augmented (with reasoning)"
        else:
            fmt = "Clean (direct answers)"
        print(f"Data format: {fmt}")
    group_datasets: Dict[frozenset, QADataset] = {}
    tok_iter = groups.items()
    if IS_MAIN:
        tok_iter = tqdm(tok_iter, desc="Tokenize", total=len(groups))
    for feat_key, qa_list in tok_iter:
        group_datasets[feat_key] = QADataset(qa_list, tokenizer,
                                             args.max_length,
                                             use_augmented=use_augmented,
                                             use_tool_call=use_tool_call,
                                             answer_value_only=args.train_answer_value_only,
                                             dynamic_padding=args.dynamic_train_padding)
    if IS_MAIN:
        total_samples = sum(len(ds) for ds in group_datasets.values())
        print(f"Total tokenized samples: {total_samples:,}")

    trainable_params = wrapper.trainable_parameters()
    weighting_params = wrapper.weighting_parameters()
    weighting_lr = float(
        getattr(args, "lora_weight_learning_rate", 0.0) or 0.0)
    if weighting_params and weighting_lr > 0:
        weighting_ids = {id(param) for param in weighting_params}
        adapter_params = [
            param for param in trainable_params
            if id(param) not in weighting_ids
        ]
        parameter_groups = []
        if adapter_params:
            parameter_groups.append({
                "params": adapter_params,
                "lr": args.learning_rate,
            })
        parameter_groups.append({
            "params": weighting_params,
            "lr": weighting_lr,
        })
        optimizer = torch.optim.AdamW(
            parameter_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=0.01,
        )
    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    _save_training_run_config(args)

    start_epoch = 0
    start_epoch_step = 0
    global_step = 0
    optimizer_step_index = 0
    latest_metrics = None

    wrapper.base_model.train()

    keys = list(group_datasets.keys())
    loaders: Dict[frozenset, DataLoader] = {}
    samplers: Dict[frozenset, DistributedSampler] = {}
    for feat_key in keys:
        ds = group_datasets[feat_key]
        if len(ds) == 0:
            continue
        sampler = DistributedSampler(
            ds,
            num_replicas=WORLD_SIZE if IS_DDP else 1,
            rank=RANK if IS_DDP else 0,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
        )
        samplers[feat_key] = sampler
        loaders[feat_key] = DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=False)

    total_batches = sum(len(loader) for loader in loaders.values())
    valid_keys = list(loaders.keys())
    scheduler, total_scheduler_steps, warmup_steps = _build_lr_scheduler(
        optimizer, args, total_batches)
    checkpoint_steps = _build_checkpoint_steps(
        total_scheduler_steps, args.num_checkpoints)
    checkpoint_step_set = set(checkpoint_steps)
    if IS_MAIN:
        print(f"LR scheduler: {args.lr_scheduler_type} | "
              f"total optimizer steps={total_scheduler_steps} | "
              f"warmup steps={warmup_steps}")
        if checkpoint_steps:
            print(f"Checkpoints: {len(checkpoint_steps)} saves at optimizer "
                  f"steps {checkpoint_steps[0]}..{checkpoint_steps[-1]}")
        else:
            print("Checkpoints: disabled")

    if args.resume_from_checkpoint:
        (start_epoch, start_epoch_step, global_step,
         optimizer_step_index) = _load_training_checkpoint(
            wrapper, optimizer, args.resume_from_checkpoint, scheduler=scheduler)
        if args.offload_optimizer_state:
            _offload_optimizer_state_to_cpu(optimizer)
        if IS_MAIN:
            print(f"[checkpoint] resumed from "
                  f"{public_path(args.resume_from_checkpoint)} "
                  f"(epoch={start_epoch}, epoch_step={start_epoch_step}, "
                  f"global_step={global_step})")

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        accum_params = {}

        for sampler in samplers.values():
            sampler.set_epoch(epoch)
        epoch_schedule = _build_group_batch_schedule(
            loaders, seed=args.seed, epoch=epoch)
        loader_iters = {k: iter(loaders[k]) for k in valid_keys}
        if epoch == start_epoch and start_epoch_step > 0:
            if IS_MAIN:
                print(f"[checkpoint] skipping {start_epoch_step} already "
                      f"completed steps in epoch {epoch + 1}")
            _advance_loader_for_resume(
                epoch_schedule, loader_iters, steps=start_epoch_step)
            epoch_steps = start_epoch_step

        pbar_range = range(epoch_steps, total_batches)
        if IS_MAIN:
            pbar = tqdm(pbar_range, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        else:
            pbar = pbar_range

        for _ in pbar:
            feat_key = epoch_schedule[epoch_steps]

            try:
                batch = next(loader_iters[feat_key])
            except StopIteration as exc:
                raise RuntimeError(
                    "Group loader exhausted before its scheduled epoch batches "
                    f"were consumed: {sorted(feat_key)}") from exc
            batch = _batch_to_device(batch, device)

            if epoch_steps % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                accum_params = {}

            active_loras = _training_active_loras(feat_key, args)
            lambda_ortho = float(getattr(args, "lora_soft_orthogonal_lambda", 0.0) or 0.0)
            batch_rows = next(iter(batch.values())).shape[0]
            micro_batch_size = int(getattr(args, "train_micro_batch_size", 0) or 0)
            if micro_batch_size <= 0 or micro_batch_size >= batch_rows:
                micro_batch_size = batch_rows

            step_total_loss = None
            step_ce_loss = None
            ortho_loss = None
            for chunk_start in range(0, batch_rows, micro_batch_size):
                chunk_end = min(chunk_start + micro_batch_size, batch_rows)
                chunk = (
                    batch if chunk_start == 0 and chunk_end == batch_rows
                    else _slice_batch(batch, chunk_start, chunk_end)
                )
                chunk_weight = float(chunk_end - chunk_start) / float(batch_rows)

                wrapper.set_active_loras(active_loras)
                out = wrapper.base_model(**chunk)
                ce_loss = out.loss
                (ce_loss * chunk_weight / grad_accum_steps).backward()

                detached_ce = ce_loss.detach() * chunk_weight
                detached_total = detached_ce
                step_total_loss = (
                    detached_total if step_total_loss is None
                    else step_total_loss + detached_total
                )
                step_ce_loss = (
                    detached_ce if step_ce_loss is None
                    else step_ce_loss + detached_ce
                )

            # The parameter-only regularizer is identical for every micro-batch.
            # Backpropagating it once is equivalent to the weighted per-chunk sum.
            if args.lora_train_stage == "knowledge_only" and lambda_ortho > 0:
                ortho_loss = wrapper.soft_orthogonality_loss(
                    args.task_lora_name, active_loras)
            if ortho_loss is not None:
                (lambda_ortho * ortho_loss / grad_accum_steps).backward()
                step_ortho_loss = ortho_loss.detach()
                step_total_loss = (
                    step_total_loss + lambda_ortho * step_ortho_loss
                )
            else:
                step_ortho_loss = torch.zeros(
                    (), device=device, dtype=step_total_loss.dtype)

            loss = step_total_loss / grad_accum_steps
            ce_loss = step_ce_loss
            ortho_loss = step_ortho_loss
            for name in active_loras:
                for p in wrapper.lora_parameters(name):
                    accum_params[id(p)] = p

            should_step = ((epoch_steps + 1) % grad_accum_steps == 0
                           or epoch_steps + 1 == total_batches)
            active_params = list(accum_params.values())
            if IS_DDP and should_step:
                for p in active_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(WORLD_SIZE)

            if should_step:
                torch.nn.utils.clip_grad_norm_(active_params, max_norm=1.0)
                if args.offload_optimizer_state:
                    _move_optimizer_state_for_params(
                        optimizer, active_params, device)
                optimizer.step()
                scheduler.step()
                optimizer_step_index += 1
                if args.offload_optimizer_state:
                    _move_optimizer_state_for_params(
                        optimizer, active_params, torch.device("cpu"))

            epoch_loss += loss.item() * grad_accum_steps
            epoch_steps += 1
            global_step += 1
            step_loss = loss.detach() * grad_accum_steps
            step_ce_loss = ce_loss.detach()
            step_ortho_loss = (
                ortho_loss.detach()
                if ortho_loss is not None
                else torch.zeros((), device=device, dtype=step_loss.dtype)
            )
            if IS_DDP:
                step_loss_tensor = step_loss.to(device=device, dtype=torch.float32)
                step_ce_tensor = step_ce_loss.to(device=device, dtype=torch.float32)
                step_ortho_tensor = step_ortho_loss.to(device=device, dtype=torch.float32)
                dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_ce_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(step_ortho_tensor, op=dist.ReduceOp.SUM)
                step_loss_value = (step_loss_tensor / WORLD_SIZE).item()
                step_ce_value = (step_ce_tensor / WORLD_SIZE).item()
                step_ortho_value = (step_ortho_tensor / WORLD_SIZE).item()
            else:
                step_loss_value = step_loss.item()
                step_ce_value = step_ce_loss.item()
                step_ortho_value = step_ortho_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            latest_metrics = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epoch": epoch + 1,
                "epoch_index": epoch,
                "epoch_step": epoch_steps,
                "global_step": global_step,
                "optimizer_step": bool(should_step),
                "optimizer_step_index": int(optimizer_step_index),
                "loss": float(step_loss_value),
                "ce_loss": float(step_ce_value),
                "soft_orthogonality_loss": float(step_ortho_value),
                "soft_orthogonality_lambda": float(lambda_ortho),
                "running_epoch_loss": float(epoch_loss / max(epoch_steps, 1)),
                "learning_rate": float(current_lr),
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_steps": int(warmup_steps),
                "active_loras": sorted(active_loras),
                "loss_function": _loss_function_description(),
            }
            weight_summary = wrapper.lora_weight_summary()
            if weight_summary is not None:
                latest_metrics["lora_weight_summary"] = weight_summary
            _write_realtime_training_state(args, latest_metrics)

            if should_step and optimizer_step_index in checkpoint_step_set:
                ckpt_idx = checkpoint_steps.index(optimizer_step_index) + 1
                ckpt_name = (
                    f"checkpoint_{ckpt_idx:03d}_step_"
                    f"{optimizer_step_index:08d}"
                )
                _save_training_checkpoint(
                    wrapper, optimizer, args, epoch=epoch,
                    epoch_step=epoch_steps, global_step=global_step,
                    latest_metrics=latest_metrics,
                    scheduler=scheduler,
                    optimizer_step_index=optimizer_step_index,
                    checkpoint_name=ckpt_name)
                if IS_DDP:
                    dist.barrier()

            if IS_MAIN:
                pbar.set_postfix(
                    loss=f"{epoch_loss / max(epoch_steps, 1):.4f}")

        if IS_DDP:
            loss_tensor = torch.tensor([epoch_loss, float(epoch_steps)], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor[0].item()
            epoch_steps = int(loss_tensor[1].item())

        elapsed = time.time() - t0
        avg = epoch_loss / max(epoch_steps, 1)
        if IS_MAIN:
            print(f"Epoch {epoch + 1} | {elapsed:.0f}s | "
                  f"avg loss {avg:.4f} | steps {epoch_steps}")

        if IS_DDP:
            dist.barrier()

    if IS_MAIN:
        wrapper.save_all_loras(os.path.join(args.output_dir, "final"))
        weight_summary = wrapper.lora_weight_summary()
        if weight_summary is not None:
            print("Residual-gate weights:")
            for dim, payload in sorted(weight_summary.items()):
                print(f"  {dim}: logit={payload['logit']:.6f}, "
                      f"weight={payload['weight']:.6f}")
        print("\nTraining complete. Adapters -> "
              f"{public_path(args.output_dir)}")
    if IS_DDP:
        dist.barrier()

# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compositional Multi-LoRA for WVS Value Survey")

    # --- data ---
    parser.add_argument(
        "--training_data_dir", type=str,
        default=None,
        help="Directory containing pre-generated training data (*_train.json files). "
             "If provided, will use this instead of building from CSV.")
    parser.add_argument(
        "--wvs_csv", type=str,
        default="data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv")
    parser.add_argument(
        "--nature_options", type=str,
        default="data/wvs/nature_options.json")
    parser.add_argument(
        "--questions_split", type=str,
        default="data_splits/wvs/questions_split.json")
    parser.add_argument(
        "--users_split", type=str,
        default="data_splits/wvs/train_users.json")
    parser.add_argument(
        "--feature_dimensions", type=str, nargs='+', default=None,
        help="Demographic dimensions used to build feature adapters. "
             "Default keeps the legacy gender age region setting.")
    parser.add_argument(
        "--raw_education_features", action='store_true',
        help="Use raw WVS Q275 education codes as edu_0..edu_8 instead of "
             "legacy edu_low/edu_medium/edu_high buckets.")
    parser.add_argument(
        "--require_complete_feature_dimensions", action='store_true',
        help="Before sampling train users, keep only users that have one "
             "valid feature for every requested --feature_dimensions entry.")
    parser.add_argument(
        "--joint_adapter_groups", action='store_true',
        help="Train each full demographic combination as one joint adapter "
             "named with sorted feature tokens joined by '__'. This provides "
             "Delta_joint references for interaction residual diagnostics.")

    # --- model ---
    parser.add_argument(
        "--model_name", type=str,
        default="models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/wvs/llama")

    # --- LoRA ---
    parser.add_argument("--lora_rank",    type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=None,
        help="Linear projection attribute names that receive every task and "
             "knowledge LoRA. The default is gate_proj up_proj down_proj.")
    parser.add_argument(
        "--lora_weight_mode", type=str, default="uniform",
        choices=["uniform", "static", "learned_static", "residual_gate"],
        help="How to combine active feature LoRAs. uniform is the legacy "
             "unweighted sum; static learns global feature weights from "
             "training loss with active-set softmax; residual_gate learns "
             "independent bounded multipliers initialized at 1.")
    parser.add_argument(
        "--lora_weight_normalize", type=str, default="sum_to_active_count",
        choices=["sum_to_active_count", "sum_to_one", "none"],
        help="Weight normalization. sum_to_active_count keeps the expected "
             "LoRA scale comparable to the original linear sum.")
    parser.add_argument(
        "--lora_composition_mode", type=str, default="sum",
        choices=[
            "sum",
            "orthogonal_projection",
            "task_knowledge_projection",
            "explicit_task_knowledge_projection",
        ],
        help="How active LoRA deltas are composed. orthogonal_projection "
             "uses low-rank Frobenius inner products to remove redundant "
             "directions according to lora_dimension_order. "
             "task_knowledge_projection adds one shared task direction plus "
             "weighted feature-residual knowledge directions. "
             "explicit_task_knowledge_projection uses a separately trained "
             "task LoRA as the shared task direction.")
    parser.add_argument(
        "--lora_dimension_order", type=str, nargs="+", default=None,
        help="Dimension order used by orthogonal_projection and deterministic "
             "adapter activation. Earlier dimensions keep their directions; "
             "later dimensions are projected to residual directions.")
    parser.add_argument(
        "--lora_orthogonal_eps", type=float, default=1e-6,
        help="Numerical epsilon for LoRA delta projection denominators.")
    parser.add_argument(
        "--lora_orthogonal_strength", type=float, default=1.0,
        help="Projection strength. 1.0 removes the full component along "
             "previous dimensions; smaller values only weaken conflicts.")
    parser.add_argument(
        "--lora_soft_orthogonal_lambda", type=float, default=0.0,
        help="Soft task/knowledge orthogonality loss weight for "
             "knowledge_only training: lambda * mean_l ||A_task A_knowledge^T||_F^2.")
    parser.add_argument(
        "--lora_task_scale", type=float, default=1.0,
        help="Scale for the shared task component in task_knowledge_projection.")
    parser.add_argument(
        "--lora_knowledge_scale", type=float, default=1.0,
        help="Scale for weighted knowledge residuals in "
             "task_knowledge_projection.")
    parser.add_argument(
        "--task_lora_name", type=str, default="task_shared",
        help="Adapter name used as the explicit shared task LoRA.")
    parser.add_argument(
        "--lora_train_stage", type=str, default="joint",
        choices=["joint", "task_only", "knowledge_only", "weighting_only"],
        help="task_only activates/trains only task_lora_name. "
             "knowledge_only loads/freezes task_lora_name and trains "
             "demographic knowledge LoRAs. weighting_only loads existing "
             "adapters and trains only their composition weights.")
    parser.add_argument(
        "--load_task_lora_dir", type=str, default=None,
        help="Directory containing task_lora_name, either the adapter directory "
             "itself or a parent final/ directory.")
    parser.add_argument(
        "--freeze_task_lora", action="store_true",
        help="Freeze task_lora_name parameters after loading. Intended for "
             "knowledge_only training.")

    # --- training ---
    parser.add_argument("--num_epochs",   type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument(
        "--train_micro_batch_size", type=int, default=0,
        help="Split each training batch into smaller forward/backward chunks "
             "to reduce activation memory. 0 disables chunking.")
    parser.add_argument(
        "--dynamic_train_padding", action="store_true",
        help="Pad training samples to the longest sequence in each batch.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_weight_learning_rate", type=float, default=0.0,
                        help="Optional separate LR for trainable LoRA weights.")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Learning-rate schedule applied per optimizer step.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup ratio used when warmup_steps is negative.")
    parser.add_argument("--warmup_steps", type=int, default=-1,
                        help="Explicit warmup optimizer steps. Negative uses warmup_ratio.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="Final LR as a ratio of learning_rate for decay schedules.")
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--max_samples_per_group", type=int, default=5000)
    parser.add_argument(
        "--max_train_users", type=int, default=0,
        help="If > 0, randomly sample this many train users before building "
             "CSV-based QA pairs. This samples users, not feature groups.")
    parser.add_argument(
        "--task_train_data_mode", type=str, default="grouped",
        choices=["grouped", "global_random"],
        help="For lora_train_stage=task_only, grouped keeps the legacy WVS "
             "feature-grouped data path, while global_random builds one "
             "demographic-agnostic task pool from globally sampled WVS QA.")
    parser.add_argument(
        "--task_user_pool", type=str, default="train",
        choices=["train", "all"],
        help="User pool for global_random task-only training. train uses the "
             "configured training-user split; all samples from every WVS "
             "respondent and is intended only for explicit diagnostics.")
    parser.add_argument(
        "--task_answer_mode", type=str, default="real",
        choices=["real", "balanced_random"],
        help="For global_random task data, real uses respondent answers; "
             "balanced_random samples a valid option label uniformly so "
             "task_shared learns the MCQ output protocol rather than WVS "
             "population preferences.")
    parser.add_argument(
        "--task_max_samples", type=int, default=0,
        help="Maximum examples for global_random task-only training. "
             "0 keeps all available examples.")
    parser.add_argument(
        "--task_balance_answers", action="store_true",
        help="Balance global_random task examples across answer labels before "
             "optional task_max_samples truncation.")
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--offload_optimizer_state", action='store_true',
                        help="Move AdamW optimizer state tensors to CPU after "
                             "each optimizer step, and move active states back "
                             "to GPU before the next step.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Deprecated; checkpoint cadence is now controlled "
                             "by --num_checkpoints.")
    parser.add_argument("--num_checkpoints", type=int, default=10,
                        help="Save this many evenly spaced resumable "
                             "checkpoints over total optimizer steps. Each "
                             "checkpoint is kept under output_dir/checkpoints/ "
                             "and checkpoints/latest is updated for resume.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to output_dir/checkpoints/latest for "
                             "resuming LoRA weights, optimizer state, and "
                             "training position.")
    parser.add_argument("--no_auto_resume", action="store_true",
                        help="Disable automatic resume from "
                             "output_dir/checkpoints/latest when it exists.")
    parser.add_argument(
        "--trainable_feature_dimensions", type=str, nargs="+", default=None,
        help="If set, only LoRA adapters whose feature dimension is listed "
             "remain trainable. Useful for sequential dimension expansion.")
    parser.add_argument(
        "--train_answer_value_only", action="store_true",
        help="Train the assistant target as only the chosen option number. "
             "The input still contains the full question/options.")

    # --- eval ---
    parser.add_argument("--eval_batch_size",  type=int, default=8)
    parser.add_argument("--max_eval_users",   type=int, default=200)
    parser.add_argument("--eval_exp3_only", action='store_true',
                        help="During evaluation, run only Exp3: LoRA-only on the test set.")
    parser.add_argument("--eval_exp123_only", action='store_true',
                        help="During evaluation, run Exp1/Exp2/Exp3 and skip Exp4.")
    parser.add_argument("--direct_answer_eval", action='store_true',
                        help="During generation evaluation, ask for only the "
                             "<answer> option number instead of reasoning plus "
                             "answer. Useful for fast Exp3-only checks.")
    parser.add_argument("--eval_choice_logprob", action='store_true',
                        help="Evaluate choices by candidate answer-text "
                             "logprob using the clean SFT prompt format and "
                             "avoid generation.")
    parser.add_argument(
        "--invalid_prediction_fallback", default="none",
        choices=("none", "choice_logprob"),
        help="For generation evaluation, score legal choices by logprob only "
             "when the generated answer cannot be parsed.")
    parser.add_argument(
        "--eval_lora_components", type=str, default="full",
        choices=["full", "task_only", "knowledge_only"],
        help="For vLLM LoRA evaluation/merging, choose which trained adapter "
             "components are merged for each user. full is task+knowledge for "
             "explicit_task_knowledge_projection, task_only merges only "
             "task_lora_name, and knowledge_only merges demographic adapters "
             "without task_lora_name.")

    # --- generation parameters ---
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--max_new_tokens", type=int, default=5000,
                        help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # --- mode ---
    parser.add_argument("--mode", type=str, default="train_eval",
                        choices=["train_eval", "train_only", "eval_only"])
    parser.add_argument("--load_lora_dir", type=str, default=None)
    parser.add_argument("--merge_for_vllm", action='store_true',
                        help="After training, merge all feature combos "
                             "into single PEFT adapters for vLLM")
    parser.add_argument("--use_vllm_eval", action='store_true',
                        help="Run the 4-experiment evaluation via vLLM "
                             "(much faster than HF generate). Will "
                             "auto-merge adapters into merged_combos/ "
                             "if that dir does not already exist.")
    parser.add_argument("--vllm_tp_size", type=int, default=1,
                        help="Tensor-parallel size for vLLM eval")
    parser.add_argument("--vllm_dp_size", type=int, default=1,
                        help="Data-parallel size (independent replicas) for "
                             "vLLM eval. dp_size * tp_size must be <= visible "
                             "GPUs. For throughput-bound eval on N>=8 GPUs, "
                             "prefer dp_size=N, tp_size=1.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM gpu_memory_utilization for eval")
    parser.add_argument("--vllm_max_model_len", type=int, default=4096,
                        help="vLLM max_model_len for eval")

    args = parser.parse_args()
    if args.lora_weight_mode == "learned_static":
        args.lora_weight_mode = "static"

    # Initialize DDP process group if launched via torchrun.
    if IS_DDP:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if IS_MAIN:
        print("=" * 70)
        print("Compositional Multi-LoRA for WVS Value Survey")
        print(f"DDP: world_size={WORLD_SIZE}, rank={RANK}, local_rank={LOCAL_RANK}")
        print("=" * 70)
        for k, v in sanitize_for_publication(vars(args)).items():
            print(f"  {k}: {v}")
        print("=" * 70)

    # ---- load data ----
    # Check if using pre-generated training data
    use_augmented = True  # Default to augmented format
    use_tool_call = False
    if args.training_data_dir:
        print("\nUsing pre-generated training data from: "
              f"{public_path(args.training_data_dir)}")
        try:
            groups, use_augmented, use_tool_call = load_pregenerated_data(args.training_data_dir)
            if args.joint_adapter_groups:
                groups = _collapse_groups_to_joint_adapters(groups)
                print(f"Joint adapter mode: collapsed to {len(groups)} "
                      "combination adapters")
        except Exception as e:
            print(f"Fatal error loading pre-generated data: {e}")
            return

        # Still need to load test data for evaluation
        if args.mode in ("train_eval", "eval_only"):
            try:
                wvs_data = load_wvs_csv(args.wvs_csv)
                nature_options = load_json(args.nature_options)
                q_split = load_json(args.questions_split)
                u_split = load_json(args.users_split)
                test_qids = [q['id'] for q in q_split['test']]
                test_uids = u_split['test']
                train_uids = u_split['train']
            except Exception as e:
                print(f"Warning: Could not load test data for evaluation: {e}")
                if args.mode == "eval_only":
                    return
                test_qids = []
                test_uids = []
                train_uids = []
    else:
        # Original data loading path
        try:
            wvs_data = load_wvs_csv(args.wvs_csv)
            nature_options = load_json(args.nature_options)
            q_split = load_json(args.questions_split)
            u_split = load_json(args.users_split)
        except Exception as e:
            print(f"Fatal error loading data files: {e}")
            return

        try:
            train_qids = [q['id'] for q in q_split['train']]
            test_qids  = [q['id'] for q in q_split['test']]
            train_uids = u_split['train']
            test_uids  = u_split['test']
        except KeyError as e:
            print(f"Error: Missing expected key in split files: {e}")
            return
        except Exception as e:
            print(f"Error processing split data: {e}")
            return

        print(f"\nTrain Q: {len(train_qids)}  |  Test Q: {len(test_qids)}")
        print(f"Train U: {len(train_uids):,}  |  Test U: {len(test_uids)}")
        use_all_task_users = (
            args.lora_train_stage == "task_only"
            and args.task_train_data_mode == "global_random"
            and args.task_user_pool == "all"
        )
        if use_all_task_users:
            train_pool_uids = list(range(len(wvs_data)))
            if IS_MAIN:
                print("Global task user pool: all WVS respondents "
                      f"({len(train_pool_uids):,})")
        else:
            train_pool_uids = _filter_complete_feature_users(
                wvs_data, train_uids, args)
        train_build_uids = _sample_train_user_ids(train_pool_uids, args)
        groups = None  # Built during training unless joint adapter names are needed now.
        if args.joint_adapter_groups and args.mode in ("train_eval", "train_only"):
            groups = build_grouped_qa(
                wvs_data, train_build_uids, train_qids, nature_options,
                max_samples_per_group=args.max_samples_per_group,
                feature_dimensions=args.feature_dimensions,
                raw_education_features=args.raw_education_features,
            )
            groups = _collapse_groups_to_joint_adapters(groups)
            use_augmented = True
            use_tool_call = False
            print(f"Joint adapter mode: built {len(groups)} "
                  "combination adapters")

    lora_names = set()
    if groups:
        for features in groups:
            lora_names.update(features)

    if (args.feature_dimensions and args.mode in ("train_eval", "eval_only")
            and 'wvs_data' in locals()):
        for idx in set(train_uids) | set(test_uids):
            if idx < len(wvs_data):
                features = get_user_features(
                    wvs_data[idx],
                    dimensions=args.feature_dimensions,
                    raw_education=args.raw_education_features,
                )
                lora_names.update(features)

    # In WVS knowledge-only training, groups are intentionally built after the
    # model to keep preprocessing out of model initialization.  Discover the
    # dynamic country_* and raw edu_* adapters from the selected training users
    # here so the wrapper creates (and can train) every adapter used by groups.
    if (args.lora_train_stage == "knowledge_only"
            and args.mode in ("train_eval", "train_only")
            and 'wvs_data' in locals()
            and 'train_build_uids' in locals()):
        lora_names.update(_collect_feature_names_for_users(
            wvs_data, train_build_uids, args))

    if not lora_names:
        lora_names = set(ALL_FEATURES)
    if (args.lora_composition_mode == "explicit_task_knowledge_projection"
            or args.lora_train_stage in ("task_only", "knowledge_only")
            or args.load_task_lora_dir):
        lora_names.add(args.task_lora_name)
    lora_names = sorted(lora_names)

    print(f"Feature dimensions: {args.feature_dimensions or ['gender', 'age', 'region']}")
    print(f"Features ({len(lora_names)}): {lora_names}")

    knowledge_trainable_names = None
    if args.lora_train_stage == "knowledge_only":
        if groups:
            knowledge_trainable_names = set()
            for features in groups:
                knowledge_trainable_names.update(features)
        elif 'wvs_data' in locals() and 'train_build_uids' in locals():
            knowledge_trainable_names = _collect_feature_names_for_users(
                wvs_data, train_build_uids, args)
        if knowledge_trainable_names is not None:
            knowledge_trainable_names.discard(args.task_lora_name)
            knowledge_trainable_names = sorted(
                name for name in knowledge_trainable_names
                if name in set(lora_names)
            )
            if IS_MAIN:
                print("Knowledge trainable adapters "
                      f"({len(knowledge_trainable_names)}): "
                      f"{knowledge_trainable_names}")

    fast_vllm_exp3_only = (
        args.mode == "eval_only"
        and args.use_vllm_eval
        and args.eval_exp3_only
    )
    fast_merge_dir = _eval_merge_dir(args)
    if fast_vllm_exp3_only and os.path.isdir(fast_merge_dir) and os.listdir(fast_merge_dir):
        evaluation_seconds = 0.0
        eval_start_time = time.time()
        print("\n" + "=" * 70)
        print("EVALUATION (vLLM Exp3-only fast path)")
        print("=" * 70)
        print("[vLLM eval] reusing existing merged adapters at "
              f"{public_path(fast_merge_dir)}")
        try:
            config, is_local = load_backbone_config(args.model_name)
            tokenizer = load_backbone_tokenizer(
                args.model_name, config=config, is_local=is_local)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            eval_test_uids = list(test_uids)
            if 0 < args.max_eval_users < len(eval_test_uids):
                rng = random.Random(args.seed)
                eval_test_uids = rng.sample(eval_test_uids, args.max_eval_users)

            from evaluate_vllm import evaluate_all_vllm
            for _k in ("RANK", "WORLD_SIZE", "LOCAL_RANK",
                       "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                       "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                       "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
                       "TORCHELASTIC_MAX_RESTARTS",
                       "TORCHELASTIC_USE_AGENT_STORE",
                       "TORCHELASTIC_ERROR_FILE"):
                os.environ.pop(_k, None)

            vllm_results = evaluate_all_vllm(
                wvs_data=wvs_data,
                user_indices=eval_test_uids,
                question_ids=test_qids,
                nature_options=nature_options,
                tokenizer=tokenizer,
                model_path=args.model_name,
                merged_lora_dir=fast_merge_dir,
                lora_rank=args.lora_rank,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.vllm_max_model_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                tp_size=args.vllm_tp_size,
                gpu_mem_util=args.vllm_gpu_memory_utilization,
                dp_size=args.vllm_dp_size,
                feature_dimensions=args.feature_dimensions,
                raw_education_features=args.raw_education_features,
                answer_value_only=args.train_answer_value_only,
                direct_answer_eval=args.direct_answer_eval,
                run_experiments=("lora_only",),
                extra_lora_count=_eval_extra_lora_count(args),
            )
            lora_only_test = vllm_results.get("lora_only", {})
            results = {'exp3_lora_only_test_set': lora_only_test}

            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

            evaluation_seconds = time.time() - eval_start_time
            timing_path = os.path.join(args.output_dir, "run_timing.json")
            with open(timing_path, "w") as f:
                json.dump({
                    "training_seconds": 0.0,
                    "evaluation_seconds": evaluation_seconds,
                }, f, indent=2, sort_keys=True)
                f.write("\n")

            print("\n" + "=" * 70)
            print("EVALUATION SUMMARY (vLLM)")
            print("=" * 70)
            print(
                f"Exp3: LoRA Only ({args.eval_lora_components}) - test set : "
                f"Acc={lora_only_test.get('accuracy', 0):.4f}, "
                f"MAE={lora_only_test.get('mae', 0):.4f}"
            )
            print("Results saved -> "
                  f"{public_path(os.path.join(args.output_dir, 'eval_results.json'))}")
            print(f"Timing saved -> {public_path(timing_path)}")
            print("\nDone!")
            return
        except Exception as e:
            print(f"Error during fast vLLM Exp3 evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
    elif fast_vllm_exp3_only:
        print("[vLLM eval] fast path unavailable: missing merged adapters at "
              f"{public_path(fast_merge_dir)}; falling back to full "
              "eval_only path.")

    # ---- load model ----
    if IS_MAIN:
        print(f"\nLoading model: {public_path(args.model_name)}")
    try:
        config, is_local = load_backbone_config(args.model_name)
        tokenizer = load_backbone_tokenizer(
            args.model_name, config=config, is_local=is_local)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Under DDP, each rank puts the full model on its own GPU.
        # Single-process mode keeps the old behavior (device_map="auto").
        if IS_DDP:
            device_map = {"": f"cuda:{LOCAL_RANK}"}
        else:
            device_map = "auto"

        model = load_backbone_model(
            args.model_name,
            config=config,
            is_local=is_local,
            device_map=device_map,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        return

    wrapper_lora_names = lora_names
    if (args.mode in ("train_eval", "train_only")
            and args.lora_train_stage == "task_only"):
        wrapper_lora_names = [args.task_lora_name]
        if IS_MAIN:
            print("Task-only wrapper: constructing only "
                  f"'{args.task_lora_name}'")

    try:
        wrapper = MultiLoRAModelWrapper(
            model, wrapper_lora_names,
            rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout, model_name=args.model_name,
        weight_mode=args.lora_weight_mode,
        weight_normalize=args.lora_weight_normalize,
            composition_mode=args.lora_composition_mode,
            dimension_order=(
                args.lora_dimension_order
                or args.feature_dimensions
                or ["gender", "age_group", "region"]
            ),
            orthogonal_eps=args.lora_orthogonal_eps,
            orthogonal_strength=args.lora_orthogonal_strength,
            task_scale=args.lora_task_scale,
            knowledge_scale=args.lora_knowledge_scale,
            task_lora_name=args.task_lora_name,
            target_modules=args.lora_target_modules,
        )
    except Exception as e:
        print(f"Fatal error creating LoRA wrapper: {e}")
        return

    if (args.mode in ("train_eval", "train_only")
            and not args.resume_from_checkpoint
            and not args.no_auto_resume):
        latest_ckpt = os.path.join(args.output_dir, "checkpoints", "latest")
        if os.path.isdir(latest_ckpt):
            args.resume_from_checkpoint = latest_ckpt
            if IS_MAIN:
                print("[checkpoint] auto-resume enabled: "
                      f"{public_path(latest_ckpt)}")

    if (args.mode in ("train_eval", "train_only")
            and args.load_lora_dir
            and not args.resume_from_checkpoint):
        if IS_MAIN:
            print("[init] loading LoRA adapters before training: "
                  f"{public_path(args.load_lora_dir)}")
        try:
            wrapper.load_all_loras(args.load_lora_dir)
        except Exception as e:
            print(f"Error loading initial LoRA adapters: {e}")
            return

    if (args.mode in ("train_eval", "train_only", "eval_only")
            and args.load_task_lora_dir
            and not args.resume_from_checkpoint):
        try:
            _load_task_lora(wrapper, args.load_task_lora_dir, args.task_lora_name)
        except Exception as e:
            print(f"Error loading task LoRA adapter: {e}")
            return

    if (args.mode in ("train_eval", "train_only")
            and args.trainable_feature_dimensions):
        wrapper.set_trainable_feature_dimensions(
            args.trainable_feature_dimensions)

    if args.mode in ("train_eval", "train_only"):
        _configure_stage_trainability(
            wrapper, args, lora_names, knowledge_trainable_names)

    # ---- train ----
    training_seconds = 0.0
    if args.mode in ("train_eval", "train_only"):
        train_start_time = time.time()
        try:
            # If using pre-generated data, groups is already loaded
            if not args.training_data_dir and groups is None:
                if 'train_build_uids' not in locals():
                    train_pool_uids = _filter_complete_feature_users(
                        wvs_data, train_uids, args)
                    train_build_uids = _sample_train_user_ids(
                        train_pool_uids, args)
                if (args.lora_train_stage == "task_only"
                        and args.task_train_data_mode == "global_random"):
                    groups = build_global_task_qa(
                        wvs_data, train_build_uids, train_qids,
                        nature_options, args)
                else:
                    groups = build_grouped_qa(
                        wvs_data, train_build_uids, train_qids, nature_options,
                        max_samples_per_group=args.max_samples_per_group,
                        feature_dimensions=args.feature_dimensions,
                        raw_education_features=args.raw_education_features,
                    )
                use_augmented = True
                if args.joint_adapter_groups:
                    groups = _collapse_groups_to_joint_adapters(groups)
            # Global task data is already bounded by task_max_samples. The
            # per-demographic cap is for grouped knowledge data; applying it
            # here would collapse the sole global task group to only a few
            # batches.
            if _should_apply_group_sample_cap(args):
                for features in groups:
                    if len(groups[features]) > args.max_samples_per_group:
                        groups[features] = groups[features][:args.max_samples_per_group]

            train_multi_lora(wrapper, groups, tokenizer, args,
                            use_augmented=use_augmented, use_tool_call=use_tool_call)
            training_seconds = time.time() - train_start_time
            del groups
            gc.collect()
        except Exception as e:
            training_seconds = time.time() - train_start_time
            print(f"Error during training: {e}", flush=True)
            if IS_DDP and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            # Training failure invalidates both train_only and train_eval.
            # Continuing to evaluation would score partially updated or
            # untrained in-memory adapters and can falsely return exit code 0.
            raise

    # ---- DDP cleanup: non-main ranks release GPU and exit.
    # The evaluation / merge / vLLM stages below are driven by rank 0 alone
    # (vLLM spawns its own TP=world_size worker pool and needs the full
    # GPU set free). Non-main ranks must drop model + process group before
    # rank 0 tries to claim their GPUs.
    if IS_DDP:
        if not IS_MAIN or args.use_vllm_eval:
            try:
                wrapper.base_model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist.barrier()
        dist.destroy_process_group()
        if not IS_MAIN:
            return

    # ---- load (eval_only) ----
    if args.mode == "eval_only":
        load_dir = args.load_lora_dir or os.path.join(args.output_dir,
                                                       "final")
        try:
            wrapper.load_all_loras(load_dir)
            # An explicitly supplied task adapter must take precedence over
            # the task adapter bundled in the full checkpoint.
            if args.load_task_lora_dir:
                _load_task_lora(
                    wrapper, args.load_task_lora_dir, args.task_lora_name)
        except Exception as e:
            print(f"Error loading LoRA adapters: {e}")
            return

    # ---- evaluate ----
    evaluation_seconds = 0.0
    if args.mode in ("train_eval", "eval_only"):
        eval_start_time = time.time()
        # Sample test users for evaluation
        eval_test_uids = list(test_uids)
        if 0 < args.max_eval_users < len(eval_test_uids):
            rng = random.Random(args.seed)
            eval_test_uids = rng.sample(eval_test_uids, args.max_eval_users)

        # Sample train users for overfitting check
        eval_train_uids = list(train_uids)
        if 0 < args.max_eval_users < len(eval_train_uids):
            rng = random.Random(args.seed)
            eval_train_uids = rng.sample(eval_train_uids, args.max_eval_users)

        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        if args.use_vllm_eval:
            try:
                merge_dir = _eval_merge_dir(args)
                vllm_run_experiments = (
                    ("lora_only",) if args.eval_exp3_only
                    else (
                        ("baseline", "persona_only", "lora_only")
                        if args.eval_exp123_only
                        else ("baseline", "persona_only", "lora_only",
                              "persona_and_lora")
                    )
                )
                needs_vllm_lora = any(
                    e in vllm_run_experiments
                    for e in ("lora_only", "persona_and_lora")
                )
                if needs_vllm_lora:
                    if not (os.path.isdir(merge_dir) and os.listdir(merge_dir)):
                        print(f"[vLLM eval] merging adapters for {len(eval_test_uids)} "
                              f"test users into {merge_dir}")
                        merge_all_combinations(
                            wrapper, wvs_data, eval_test_uids, merge_dir,
                            feature_dimensions=args.feature_dimensions,
                            raw_education_features=args.raw_education_features,
                            joint_adapter_groups=args.joint_adapter_groups,
                            extra_active_loras=_eval_extra_active_loras(args),
                            lora_component_mode=args.eval_lora_components,
                        )
                    else:
                        print("[vLLM eval] reusing existing merged adapters at "
                              f"{public_path(merge_dir)}")

                # Free GPU for vLLM. Qwen3.5 is loaded through Accelerate's
                # device map, where .to("cpu") can be rejected and leave the
                # full HF model resident. Once adapters are merged, the
                # vLLM path no longer needs the wrapper/model.
                try:
                    wrapper.base_model.to('cpu')
                except Exception as exc:
                    print(f"[vLLM eval] HF CPU offload was unavailable: {exc}")
                if not args.merge_for_vllm:
                    wrapper = None
                    model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                from evaluate_vllm import evaluate_all_vllm
                # Strip torchrun-injected env vars so vLLM's internal
                # rendezvous (TCPStore for TP / DP workers) is not derailed
                # by a stale MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE.
                for _k in ("RANK", "WORLD_SIZE", "LOCAL_RANK",
                           "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                           "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                           "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
                           "TORCHELASTIC_MAX_RESTARTS",
                           "TORCHELASTIC_USE_AGENT_STORE",
                           "TORCHELASTIC_ERROR_FILE"):
                    os.environ.pop(_k, None)
                vllm_results = {}
                if vllm_run_experiments:
                    vllm_results = evaluate_all_vllm(
                        wvs_data=wvs_data,
                        user_indices=eval_test_uids,
                        question_ids=test_qids,
                        nature_options=nature_options,
                        tokenizer=tokenizer,
                        model_path=args.model_name,
                        merged_lora_dir=merge_dir,
                        lora_rank=args.lora_rank,
                        max_new_tokens=args.max_new_tokens,
                        max_model_len=args.vllm_max_model_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        tp_size=args.vllm_tp_size,
                        gpu_mem_util=args.vllm_gpu_memory_utilization,
                        dp_size=args.vllm_dp_size,
                        feature_dimensions=args.feature_dimensions,
                        raw_education_features=args.raw_education_features,
                        answer_value_only=args.train_answer_value_only,
                        direct_answer_eval=args.direct_answer_eval,
                        run_experiments=vllm_run_experiments,
                        extra_lora_count=_eval_extra_lora_count(args),
                    )
                baseline = vllm_results.get("baseline", {})
                persona_prompt_only = vllm_results.get("persona_only", {})
                lora_only_test = vllm_results.get("lora_only", {})
                prompt_and_lora_test = vllm_results.get("persona_and_lora", {})

                if args.eval_exp3_only:
                    results = {'exp3_lora_only_test_set': lora_only_test}
                elif args.eval_exp123_only:
                    results = {
                        'exp1_baseline_test_set': baseline,
                        'exp2_prompt_only_test_set': persona_prompt_only,
                        'exp3_lora_only_test_set': lora_only_test,
                    }
                else:
                    results = {
                        'exp1_baseline_test_set': baseline,
                        'exp2_prompt_only_test_set': persona_prompt_only,
                        'exp3_lora_only_test_set': lora_only_test,
                        'exp4_prompt_and_lora_test_set': prompt_and_lora_test,
                    }
                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(os.path.join(args.output_dir, 'eval_results.json'),
                              'w') as f:
                        json.dump(results, f, indent=2)
                except Exception as e:
                    print(f"Warning: Error saving evaluation results: {e}")

                print("\n" + "=" * 70)
                print("EVALUATION SUMMARY (vLLM)")
                print("=" * 70)
                def _fmt(m):
                    if not m:
                        return "(skipped)"
                    return f"Acc={m.get('accuracy', 0):.4f}, MAE={m.get('mae', 0):.4f}"
                if not args.eval_exp3_only:
                    print(f"Exp1: Baseline (no prompt, no LoRA)    : {_fmt(baseline)}")
                    print(f"Exp2: Prompt Only (no LoRA)            : {_fmt(persona_prompt_only)}")
                print(f"Exp3: LoRA Only (no prompt) - test set : {_fmt(lora_only_test)}")
                if not args.eval_exp3_only and not args.eval_exp123_only:
                    print(f"Exp4: Prompt + LoRA - test set         : {_fmt(prompt_and_lora_test)}")
                print()
            except Exception as e:
                print(f"Error during vLLM evaluation: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                baseline = {}
                persona_prompt_only = {}
                prompt_and_lora_test = {}
                if not args.eval_exp3_only:
                    # Evaluation 1: Baseline (no prompt, no LoRA) on test set
                    baseline = evaluate(
                        wrapper, wvs_data, eval_test_uids, test_qids,
                        nature_options, tokenizer, args,
                        desc="Exp1: Baseline (no prompt, no LoRA) – test set", use_lora=False)

                    # Evaluation 2: Persona prompt only (no LoRA) on test set
                    persona_prompt_only = evaluate_with_persona_prompt(
                        wrapper, wvs_data, eval_test_uids, test_qids,
                        nature_options, tokenizer, args,
                        desc="Exp2: Persona Prompt Only (no LoRA) – test set", use_lora=False)

                # Evaluation 3: Multi-LoRA only (no prompt) on test set
                if args.eval_choice_logprob:
                    lora_only_test = evaluate_choice_logprob(
                        wrapper, wvs_data, eval_test_uids, test_qids,
                        nature_options, tokenizer, args,
                        desc="Exp3: Multi-LoRA Choice Logprob – test set",
                        use_lora=True)
                else:
                    lora_only_test = evaluate(
                        wrapper, wvs_data, eval_test_uids, test_qids,
                        nature_options, tokenizer, args,
                        desc="Exp3: Multi-LoRA Only (no prompt) – test set", use_lora=True)

                if not args.eval_exp3_only and not args.eval_exp123_only:
                    # Evaluation 4: Persona prompt + Multi-LoRA on test set
                    prompt_and_lora_test = evaluate_with_persona_prompt(
                        wrapper, wvs_data, eval_test_uids, test_qids,
                        nature_options, tokenizer, args,
                        desc="Exp4: Persona Prompt + Multi-LoRA – test set", use_lora=True)

                if args.eval_exp3_only:
                    results = {'exp3_lora_only_test_set': lora_only_test}
                elif args.eval_exp123_only:
                    results = {
                        'exp1_baseline_test_set': baseline,
                        'exp2_prompt_only_test_set': persona_prompt_only,
                        'exp3_lora_only_test_set': lora_only_test,
                    }
                else:
                    results = {
                        'exp1_baseline_test_set': baseline,
                        'exp2_prompt_only_test_set': persona_prompt_only,
                        'exp3_lora_only_test_set': lora_only_test,
                        'exp4_prompt_and_lora_test_set': prompt_and_lora_test,
                    }

                try:
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(os.path.join(args.output_dir, 'eval_results.json'),
                              'w') as f:
                        json.dump(results, f, indent=2)
                except Exception as e:
                    print(f"Warning: Error saving evaluation results: {e}")

                print("\n" + "=" * 70)
                print("EVALUATION SUMMARY")
                print("=" * 70)
                if not args.eval_exp3_only:
                    print(f"Exp1: Baseline (no prompt, no LoRA)    : Acc={baseline['accuracy']:.4f}, MAE={baseline['mae']:.4f}")
                    print(f"Exp2: Prompt Only (no LoRA)            : Acc={persona_prompt_only['accuracy']:.4f}, MAE={persona_prompt_only['mae']:.4f}")
                print(f"Exp3: LoRA Only (no prompt) - test set : Acc={lora_only_test['accuracy']:.4f}, MAE={lora_only_test['mae']:.4f}")
                if not args.eval_exp3_only and not args.eval_exp123_only:
                    print(f"Exp4: Prompt + LoRA - test set         : Acc={prompt_and_lora_test['accuracy']:.4f}, MAE={prompt_and_lora_test['mae']:.4f}")
                print()


            except Exception as e:
                print(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()

        evaluation_seconds = time.time() - eval_start_time
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            timing_path = os.path.join(args.output_dir, "run_timing.json")
            with open(timing_path, "w") as f:
                json.dump({
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                }, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"Timing saved -> {public_path(timing_path)}")
        except Exception as e:
            print(f"Warning: Error saving timing: {e}")

    # ---- optional: merge for vLLM ----
    if args.merge_for_vllm:
        try:
            merge_dir = _eval_merge_dir(args)
            merge_all_combinations(
                wrapper, wvs_data, test_uids, merge_dir,
                feature_dimensions=args.feature_dimensions,
                raw_education_features=args.raw_education_features,
                joint_adapter_groups=args.joint_adapter_groups,
                extra_active_loras=_eval_extra_active_loras(args),
                lora_component_mode=args.eval_lora_components,
            )
        except Exception as e:
            print(f"Error during LoRA merging: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
