"""vLLM-based evaluation for compositional Multi-LoRA WVS.

Replaces the HF `generate(...)` loop in evaluate.py with a single vLLM engine
that continuously batches across all prompts. LoRA combinations are loaded via
LoRARequest, so the same engine handles baseline (no LoRA) and per-group LoRA
runs without reloading the base model.

When dp_size > 1, multiple independent vLLM engines are spawned (one per GPU)
via multiprocessing, and LoRA groups are distributed across them for parallel
throughput.

Output dict shape matches evaluate.evaluate() / evaluate_with_persona_prompt()
so train_eval.py can swap implementations transparently.
"""

import os
import gc
import multiprocessing as mp
import queue
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from config import *  # noqa: F401,F403
from distribution_metrics import compute_distribution_metrics
from utils import (
    get_user_features,
    get_valid_option_set,
    extract_answer_number,
    generate_persona_prompt,
)


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


# ----------------------------------------------------------------------
# Prompt builders (kept character-for-character in sync with evaluate.py)
# ----------------------------------------------------------------------

def _build_plain_prompt(q_text: str, opts: dict, *,
                        answer_value_only: bool = False,
                        direct_answer_eval: bool = False) -> str:
    opts_fmt = "\n".join(
        f"{k}. {v}" for k, v in sorted(
            opts.items(),
            key=lambda x: (int(x[0]) if x[0].lstrip('-').isdigit() else 0),
        )
    )
    if answer_value_only:
        return (f"Question: {q_text}\n"
                f"Options:\n{opts_fmt}\n\n"
                f"Write ONLY your chosen option number.\n\n"
                f"Answer:")
    if direct_answer_eval:
        return (f"Question: {q_text}\n"
                f"Options:\n{opts_fmt}\n\n"
                f"Write ONLY your chosen option number inside "
                f"<answer></answer> tags.\n\n"
                f"Required format:\n"
                f"<answer>[option number]</answer>\n\n"
                f"Your response:")
    return (f"Question: {q_text}\n"
            f"Options:\n{opts_fmt}\n\n"
            f"First, provide your reasoning and explanation. "
            f"Then, on a new line, write ONLY your chosen option number inside <answer></answer> tags.\n\n"
            f"Required format:\n"
            f"[Your reasoning and explanation]\n"
            f"<answer>[option number]</answer>\n\n"
            f"Your response:")


def _build_plain_messages(q_text: str, opts: dict, *,
                          answer_value_only: bool = False,
                          direct_answer_eval: bool = False):
    user_content = _build_plain_prompt(
        q_text, opts,
        answer_value_only=answer_value_only,
        direct_answer_eval=direct_answer_eval,
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]


def _build_persona_messages(q_text: str, opts: dict, profile_str: str, *,
                            answer_value_only: bool = False,
                            direct_answer_eval: bool = False):
    opts_fmt = "\n".join(
        f"{k}. {v}" for k, v in sorted(
            opts.items(),
            key=lambda x: (int(x[0]) if x[0].lstrip('-').isdigit() else 0),
        )
    )
    sys_prompt = (f"You are a person with the following profile: {profile_str}. "
                  f"You are a helpful assistant that answers survey questions honestly.")
    if answer_value_only:
        user_prompt = (f"Question: {q_text}\n"
                       f"Options:\n{opts_fmt}\n\n"
                       f"Write ONLY your chosen option number.\n\n"
                       f"Answer:")
    elif direct_answer_eval:
        user_prompt = (f"Question: {q_text}\n"
                       f"Options:\n{opts_fmt}\n\n"
                       f"Write ONLY your chosen option number inside "
                       f"<answer></answer> tags.\n\n"
                       f"Required format:\n"
                       f"<answer>[option number]</answer>\n\n"
                       f"Your response:")
    else:
        user_prompt = (f"Question: {q_text}\n"
                       f"Options:\n{opts_fmt}\n\n"
                       f"First, provide your reasoning and explanation. "
                       f"Then, on a new line, write ONLY your chosen option number inside <answer></answer> tags.\n\n"
                       f"Required format:\n"
                       f"[Your reasoning and explanation]\n"
                       f"<answer>[option number]</answer>\n\n"
                       f"Your response:")
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------
# Collect per-user / per-question items
# ----------------------------------------------------------------------

def _collect_items(wvs_data, user_indices, question_ids, nature_options,
                   feature_dimensions=None, raw_education_features=False):
    """Return per-user/per-question records used by every vLLM experiment."""
    items = []
    for idx in sorted(set(user_indices)):
        if idx >= len(wvs_data):
            continue
        row = wvs_data[idx]
        try:
            features = get_user_features(
                row, dimensions=feature_dimensions,
                raw_education=raw_education_features)
        except Exception:
            continue
        if not features:
            continue
        for qid in question_ids:
            if qid not in nature_options:
                continue
            val = row.get(qid, '')
            if not val:
                continue
            try:
                gt = int(float(val))
            except (ValueError, TypeError):
                continue
            if gt < 0:
                continue
            valid_options = get_valid_option_set(qid, nature_options)
            if valid_options is not None and gt not in valid_options:
                continue
            q_info = nature_options[qid]
            items.append({
                'participant_id': idx,
                'features': features,
                'qid': qid,
                'q_text': q_info['question_text'],
                'opts': q_info.get('options', {}),
                'gt': gt,
                'valid_opts': valid_options,
            })
    return items


# ----------------------------------------------------------------------
# vLLM engine lifecycle
# ----------------------------------------------------------------------

_VLLM_SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def _multi_lora_batch_slots() -> int:
    """Number of distinct LoRAs vLLM may schedule in one batch."""
    raw = os.environ.get("VLLM_MULTI_LORA_BATCH_SLOTS", "1")
    try:
        return max(int(raw), 1)
    except ValueError:
        raise ValueError(
            "VLLM_MULTI_LORA_BATCH_SLOTS must be a positive integer, "
            f"got {raw!r}")


def _round_up_vllm_lora_rank(rank: int) -> int:
    """Return the smallest vLLM-supported LoRA rank that can hold rank."""
    for supported_rank in _VLLM_SUPPORTED_LORA_RANKS:
        if rank <= supported_rank:
            return supported_rank
    raise ValueError(
        f"Required LoRA rank {rank} exceeds vLLM maximum supported rank "
        f"{_VLLM_SUPPORTED_LORA_RANKS[-1]}")


def _prepare_vllm_process_env():
    # train_eval.py touches CUDA during module initialization. vLLM's default
    # forked engine process cannot reinitialize CUDA after that, so force spawn.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _engine_kwargs(model_path: str, unique_feature_keys, lora_rank: int,
                   tp_size: int, gpu_mem_util: float, max_model_len: int,
                   extra_lora_count: int = 0):
    """Build kwargs dict for vllm.LLM (shared between single and DP modes)."""
    max_lora_rank = 0
    for fk in unique_feature_keys:
        merged_rank = max(len(fk) + int(extra_lora_count), 1) * lora_rank
        if merged_rank > max_lora_rank:
            max_lora_rank = merged_rank
    max_lora_rank = max(max_lora_rank, lora_rank)
    total_loras = max(len(unique_feature_keys), 1)
    # The default keeps the established one-adapter-at-a-time behavior. Large
    # evaluation GPUs can opt into scheduling several demographic adapters in
    # one request via VLLM_MULTI_LORA_BATCH_SLOTS.
    max_loras = min(_multi_lora_batch_slots(), total_loras)
    vllm_max_lora_rank = _round_up_vllm_lora_rank(max_lora_rank)
    kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        enable_lora=True,
        max_loras=max_loras,
        max_cpu_loras=total_loras,
        max_lora_rank=vllm_max_lora_rank,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    if "ministral" in os.path.basename(os.path.normpath(model_path)).lower():
        # Ministral-3 packages a Pixtral-capable processor even for text-only
        # evaluation. Prevent vLLM from profiling a dummy image at startup.
        kwargs.update(
            language_model_only=True,
            limit_mm_per_prompt={"image": 0},
        )
    return kwargs, max_loras, total_loras, vllm_max_lora_rank


def _start_engine(model_path: str, merged_lora_dir: str,
                  unique_feature_keys, lora_rank: int,
                  tp_size: int, gpu_mem_util: float, max_model_len: int,
                  dp_size: int = 1, extra_lora_count: int = 0):
    """Start a single-process vLLM LLM engine (dp_size is ignored here;
    DP is handled at a higher level via multiprocessing)."""
    _prepare_vllm_process_env()
    from vllm import LLM

    kwargs, max_loras, total_loras, max_lora_rank = _engine_kwargs(
        model_path, unique_feature_keys, lora_rank,
        tp_size, gpu_mem_util, max_model_len,
        extra_lora_count=extra_lora_count)

    print(f"[vLLM] launching engine: tp={tp_size}, "
          f"max_loras={max_loras}, max_cpu_loras={total_loras}, "
          f"max_lora_rank={max_lora_rank}, "
          f"max_model_len={max_model_len}")
    llm = LLM(**kwargs)
    return llm


# ----------------------------------------------------------------------
# Data-parallel worker (multiprocessing)
# ----------------------------------------------------------------------

def _dp_worker(rank: int, gpu_id: str, llm_kwargs: dict,
               sampling_params_dict: dict,
               groups_shard: List[Tuple],
               all_prompts: List[str],
               use_lora: bool, merged_lora_dir: str,
               lora_id_map: dict,
               result_queue: mp.Queue):
    """Worker process: create an LLM engine on one GPU, run assigned groups."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _prepare_vllm_process_env()
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(**llm_kwargs)
        sp = SamplingParams(**sampling_params_dict)

        results = {}  # original_index -> response_str
        for features, idxs in groups_shard:
            batch_prompts = [all_prompts[i] for i in idxs]
            if use_lora:
                from vllm.lora.request import LoRARequest
                name = _feature_key_str(features)
                path = os.path.join(merged_lora_dir, name)
                lora_req = LoRARequest(name, lora_id_map[features], path)
                outputs = llm.generate(batch_prompts, sp,
                                       lora_request=lora_req, use_tqdm=False)
            else:
                outputs = llm.generate(batch_prompts, sp, use_tqdm=False)
            for i, out in zip(idxs, outputs):
                results[i] = _generation_payload(out)

        result_queue.put(("ok", rank, results))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _feature_key_str(features) -> str:
    return "__".join(sorted(features))


def _generation_payload(output) -> dict:
    completion = output.outputs[0] if output.outputs else None
    return {
        "text": completion.text if completion is not None else "",
        "input_tokens": len(output.prompt_token_ids or []),
        "output_tokens": len(completion.token_ids or []) if completion is not None else 0,
    }


def _summarize_counts(values: list[int]) -> dict:
    return {
        "samples": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
    }


def _lora_request(merged_lora_dir: str, features, lora_id_map: dict):
    from vllm.lora.request import LoRARequest
    name = _feature_key_str(features)
    path = os.path.join(merged_lora_dir, name)
    return LoRARequest(name, lora_id_map[features], path)


# ----------------------------------------------------------------------
# Core generation: group prompts by LoRA slot and call llm.generate once per group
# ----------------------------------------------------------------------

def _run_generation(llm, sampling_params, items, prompts,
                    use_lora: bool, merged_lora_dir: str,
                    lora_id_map: dict, desc: str):
    """Group `items` by feature_key, run one llm.generate per group.

    Returns response payloads aligned with `items` / `prompts`.
    """
    responses = [None] * len(items)
    if use_lora and _multi_lora_batch_slots() > 1:
        lora_requests = [
            _lora_request(merged_lora_dir, it['features'], lora_id_map)
            for it in items
        ]
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_requests,
            use_tqdm=True)
        for i, out in enumerate(outputs):
            responses[i] = _generation_payload(out)
        return responses

    groups = defaultdict(list)  # features -> list of original indices
    for i, it in enumerate(items):
        groups[it['features']].append(i)

    group_iter = sorted(groups.items(), key=lambda x: _feature_key_str(x[0]))
    for features, idxs in tqdm(group_iter, desc=desc):
        batch_prompts = [prompts[i] for i in idxs]
        if use_lora:
            lora_req = _lora_request(merged_lora_dir, features, lora_id_map)
            outputs = llm.generate(batch_prompts, sampling_params,
                                   lora_request=lora_req, use_tqdm=False)
        else:
            outputs = llm.generate(batch_prompts, sampling_params,
                                   use_tqdm=False)
        for i, out in zip(idxs, outputs):
            responses[i] = _generation_payload(out)
    return responses


def _run_generation_dp(dp_size: int, gpu_ids: List[str], llm_kwargs: dict,
                       sampling_params_dict: dict,
                       items, prompts,
                       use_lora: bool, merged_lora_dir: str,
                       lora_id_map: dict, desc: str):
    """Data-parallel generation: split LoRA groups across dp_size workers.

    Each worker gets a subset of groups, creates its own LLM engine on a
    dedicated GPU, runs generation, and returns results via a Queue.
    """
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it['features']].append(i)
    group_list = sorted(groups.items(), key=lambda x: _feature_key_str(x[0]))

    # Distribute groups round-robin across workers (balance by item count)
    shards: List[List[Tuple]] = [[] for _ in range(dp_size)]
    shard_sizes = [0] * dp_size
    for g in sorted(group_list, key=lambda x: -len(x[1])):
        smallest = min(range(dp_size), key=lambda i: shard_sizes[i])
        shards[smallest].append(g)
        shard_sizes[smallest] += len(g[1])

    total_items = sum(shard_sizes)
    print(f"[vLLM DP] {desc}: {len(group_list)} groups -> {dp_size} workers "
          f"({total_items} items, shard sizes: {shard_sizes})")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    workers = []
    for rank in range(dp_size):
        if not shards[rank]:
            continue
        p = ctx.Process(
            target=_dp_worker,
            args=(rank, gpu_ids[rank], llm_kwargs, sampling_params_dict,
                  shards[rank], prompts, use_lora, merged_lora_dir,
                  lora_id_map, result_queue),
        )
        p.start()
        workers.append(p)

    responses = [None] * len(items)
    received = 0
    while received < len(workers):
        try:
            status, rank, payload = result_queue.get(timeout=5)
        except queue.Empty:
            failed = [p.exitcode for p in workers
                      if p.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(
                    f"vLLM DP worker exited before returning results; "
                    f"exit codes: {failed}")
            continue
        received += 1
        if status == "error":
            raise RuntimeError(f"vLLM DP worker {rank} failed:\n{payload}")
        for idx, text in payload.items():
            responses[idx] = text

    for p in workers:
        p.join()

    return responses


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def _score(items, responses, *, use_lora: bool, persona_str_by_features=None):
    correct = 0
    total = 0
    invalid_predictions = 0
    error_sum = 0.0
    samples_by_features: Dict[tuple, list] = {}
    invalid_outputs = []
    distribution_records = []
    input_token_counts = []
    output_token_counts = []

    for it, response in zip(items, responses):
        if isinstance(response, dict):
            input_token_counts.append(int(response.get('input_tokens', 0)))
            output_token_counts.append(int(response.get('output_tokens', 0)))
            response = response.get('text', '')
        else:
            input_token_counts.append(0)
            output_token_counts.append(0)
        response = response or ""
        features_key = tuple(sorted(it['features']))
        if features_key not in samples_by_features:
            samples_by_features[features_key] = []

        prediction = extract_answer_number(response, valid_options=it['valid_opts'])
        distribution_records.append({
            'participant_id': it['participant_id'],
            'domain': 'wvs',
            'question_id': it['qid'],
            'target': it['gt'],
            'prediction': prediction,
            'valid_options': sorted(it['valid_opts']) if it['valid_opts'] else [],
        })

        if len(samples_by_features[features_key]) < 5:
            sample = {
                'question_id': it['qid'],
                'raw_output': response,
                'extracted_prediction': prediction,
                'ground_truth': it['gt'],
                'is_correct': (prediction == it['gt']) if prediction is not None else None,
                'features': list(it['features']) if use_lora else [],
            }
            if persona_str_by_features is not None:
                sample['persona_prompt'] = persona_str_by_features.get(it['features'], '')
            samples_by_features[features_key].append(sample)

        if prediction is not None:
            abs_error = abs(prediction - it['gt'])
            num_options = len(it['valid_opts']) if it['valid_opts'] else 1
            normalized_error = abs_error / max(num_options - 1, 1)
            error_sum += normalized_error
            if prediction == it['gt']:
                correct += 1
            total += 1
        else:
            invalid_predictions += 1
            total += 1
            invalid_sample = {
                'question_id': it['qid'],
                'raw_output': response,
                'extracted_prediction': prediction,
                'ground_truth': it['gt'],
                'valid_options': sorted(it['valid_opts']) if it['valid_opts'] else None,
                'features': list(it['features']) if use_lora else [],
            }
            if persona_str_by_features is not None:
                invalid_sample['persona_prompt'] = persona_str_by_features.get(
                    it['features'], '')
            invalid_outputs.append(invalid_sample)

    acc = correct / total if total > 0 else 0.0
    valid_rate = (total - invalid_predictions) / total if total > 0 else 0.0
    valid_count = total - invalid_predictions
    mae = error_sum / valid_count if valid_count > 0 else 0.0

    sample_outputs = {
        '_'.join(sorted(features)): samples
        for features, samples in samples_by_features.items()
    }
    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'invalid_predictions': invalid_predictions,
        'valid_rate': valid_rate,
        'mae': mae,
        'valid_count': valid_count,
        'sample_outputs': sample_outputs,
        'prediction_records': distribution_records,
        'invalid_outputs': invalid_outputs,
        'token_usage': {
            'input_tokens': _summarize_counts(input_token_counts),
            'output_tokens': _summarize_counts(output_token_counts),
            'total_tokens': sum(input_token_counts) + sum(output_token_counts),
        },
        **compute_distribution_metrics(distribution_records),
    }


# ----------------------------------------------------------------------
# Top-level: run all 4 experiments with a single engine or DP pool
# ----------------------------------------------------------------------

def evaluate_all_vllm(
    *,
    wvs_data: list,
    user_indices,
    question_ids: list,
    nature_options: dict,
    tokenizer,
    model_path: str,
    merged_lora_dir: str,
    lora_rank: int,
    max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tp_size: int,
    gpu_mem_util: float,
    dp_size: int = 1,
    feature_dimensions=None,
    raw_education_features: bool = False,
    answer_value_only: bool = False,
    direct_answer_eval: bool = False,
    run_experiments: Tuple[str, ...] = ("baseline", "persona_only",
                                         "lora_only", "persona_and_lora"),
    extra_lora_count: int = 0,
) -> Dict[str, dict]:
    """Run the four eval experiments through one vLLM engine (or DP pool).

    Returns {exp_name: metrics_dict}.
    """
    from vllm import SamplingParams

    items = _collect_items(
        wvs_data, user_indices, question_ids, nature_options,
        feature_dimensions=feature_dimensions,
        raw_education_features=raw_education_features)
    if not items:
        empty = {'accuracy': 0.0, 'correct': 0, 'total': 0,
                 'invalid_predictions': 0, 'valid_rate': 0.0, 'mae': 0.0,
                 'valid_count': 0, 'sample_outputs': {},
                 'prediction_records': [],
                 'invalid_outputs': []}
        return {e: empty for e in run_experiments}
    print(f"[vLLM eval] collected {len(items)} (user, question) items "
          f"across {len(set(it['features'] for it in items))} groups")

    # Unique feature keys drive LoRA slot count.
    unique_features = sorted({it['features'] for it in items},
                             key=lambda f: _feature_key_str(f))
    lora_id_map = {fk: i + 1 for i, fk in enumerate(unique_features)}

    # Build prompts per experiment (kept ahead of engine launch so we fail fast).
    plain_prompts: List[str] = []
    persona_prompts: List[str] = []
    persona_str_by_features: Dict[frozenset, str] = {}
    need_persona = ("persona_only" in run_experiments
                    or "persona_and_lora" in run_experiments)
    need_plain = ("baseline" in run_experiments
                  or "lora_only" in run_experiments)

    has_chat_template = (hasattr(tokenizer, 'apply_chat_template')
                         and tokenizer.chat_template)

    for it in items:
        if need_plain:
            msgs = _build_plain_messages(
                it['q_text'], it['opts'],
                answer_value_only=answer_value_only,
                direct_answer_eval=direct_answer_eval,
            )
            if has_chat_template:
                plain_prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            else:
                plain_prompts.append(_build_plain_prompt(
                    it['q_text'], it['opts'],
                    answer_value_only=answer_value_only,
                    direct_answer_eval=direct_answer_eval,
                ))
        if need_persona:
            fk = it['features']
            if fk not in persona_str_by_features:
                persona_str_by_features[fk] = generate_persona_prompt(fk)
            profile_str = persona_str_by_features[fk]
            msgs = _build_persona_messages(
                it['q_text'], it['opts'], profile_str,
                answer_value_only=answer_value_only,
                direct_answer_eval=direct_answer_eval,
            )
            if has_chat_template:
                persona_prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            else:
                persona_prompts.append(f"{msgs[0]['content']}\n\n{msgs[1]['content']}")

    if need_plain and plain_prompts:
        print("\n" + "=" * 70)
        print("[DEBUG] First 3 plain prompts sent to vLLM:")
        print("=" * 70)
        for i, p in enumerate(plain_prompts[:3]):
            print(f"\n--- Prompt {i} ---")
            print(repr(p))
        print("=" * 70 + "\n")

    sampling_params_dict = dict(
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
        top_k=top_k if temperature > 0 else -1,
        max_tokens=max_new_tokens,
        n=1,
    )

    llm = None

    # --- DP path: multiprocessing, each worker gets its own GPU + engine ---
    if dp_size > 1:
        kwargs, max_loras, total_loras, max_lora_rank = _engine_kwargs(
            model_path, unique_features, lora_rank,
            tp_size, gpu_mem_util, max_model_len,
            extra_lora_count=extra_lora_count)
        print(f"[vLLM] launching DP pool: dp={dp_size}, tp={tp_size}, "
              f"max_loras={max_loras}, max_cpu_loras={total_loras}, "
              f"max_lora_rank={max_lora_rank}, "
              f"max_model_len={max_model_len}")

        # Determine per-worker GPU assignment from CUDA_VISIBLE_DEVICES
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpu_ids = visible.split(",")
        else:
            gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
        needed = dp_size * tp_size
        if len(gpu_ids) < needed:
            raise RuntimeError(
                f"dp_size={dp_size} * tp_size={tp_size} = {needed} GPUs needed, "
                f"but only {len(gpu_ids)} visible: {gpu_ids}")
        # Each worker gets tp_size consecutive GPUs
        worker_gpu_ids = []
        for w in range(dp_size):
            start = w * tp_size
            worker_gpu_ids.append(",".join(gpu_ids[start:start + tp_size]))

        def _dp_gen(prompts, use_lora, desc):
            return _run_generation_dp(
                dp_size=dp_size, gpu_ids=worker_gpu_ids,
                llm_kwargs=kwargs, sampling_params_dict=sampling_params_dict,
                items=items, prompts=prompts,
                use_lora=use_lora, merged_lora_dir=merged_lora_dir,
                lora_id_map=lora_id_map, desc=desc)
    else:
        # --- Single-engine path (dp_size == 1) ---
        llm = _start_engine(
            model_path=model_path,
            merged_lora_dir=merged_lora_dir,
            unique_feature_keys=unique_features,
            lora_rank=lora_rank,
            tp_size=tp_size,
            gpu_mem_util=gpu_mem_util,
            max_model_len=max_model_len,
            extra_lora_count=extra_lora_count,
        )
        sampling_params = SamplingParams(**sampling_params_dict)

        def _dp_gen(prompts, use_lora, desc):
            return _run_generation(
                llm, sampling_params, items, prompts,
                use_lora=use_lora, merged_lora_dir=merged_lora_dir,
                lora_id_map=lora_id_map, desc=desc)

    results: Dict[str, dict] = {}
    try:
        if "baseline" in run_experiments:
            t0 = time.time()
            resp = _dp_gen(plain_prompts, False,
                           "Exp1: Baseline (no prompt, no LoRA)")
            results["baseline"] = _score(items, resp, use_lora=False)
            results["baseline"]["runtime_seconds"] = time.time() - t0
            print(f"  Exp1 Baseline: acc={results['baseline']['accuracy']:.4f} "
                  f"mae={results['baseline']['mae']:.4f} "
                  f"time={results['baseline']['runtime_seconds']:.1f}s")

        if "persona_only" in run_experiments:
            t0 = time.time()
            resp = _dp_gen(persona_prompts, False,
                           "Exp2: Persona Prompt Only (no LoRA)")
            results["persona_only"] = _score(
                items, resp, use_lora=False,
                persona_str_by_features=persona_str_by_features)
            results["persona_only"]["runtime_seconds"] = time.time() - t0
            print(f"  Exp2 Persona only: acc={results['persona_only']['accuracy']:.4f} "
                  f"mae={results['persona_only']['mae']:.4f} "
                  f"time={results['persona_only']['runtime_seconds']:.1f}s")

        if "lora_only" in run_experiments:
            t0 = time.time()
            resp = _dp_gen(plain_prompts, True,
                           "Exp3: Multi-LoRA Only (no prompt)")
            results["lora_only"] = _score(items, resp, use_lora=True)
            results["lora_only"]["runtime_seconds"] = time.time() - t0
            print(f"  Exp3 LoRA only: acc={results['lora_only']['accuracy']:.4f} "
                  f"mae={results['lora_only']['mae']:.4f} "
                  f"time={results['lora_only']['runtime_seconds']:.1f}s")

        if "persona_and_lora" in run_experiments:
            t0 = time.time()
            resp = _dp_gen(persona_prompts, True,
                           "Exp4: Persona Prompt + Multi-LoRA")
            results["persona_and_lora"] = _score(
                items, resp, use_lora=True,
                persona_str_by_features=persona_str_by_features)
            results["persona_and_lora"]["runtime_seconds"] = time.time() - t0
            print(f"  Exp4 Persona + LoRA: acc={results['persona_and_lora']['accuracy']:.4f} "
                  f"mae={results['persona_and_lora']['mae']:.4f} "
                  f"time={results['persona_and_lora']['runtime_seconds']:.1f}s")
    finally:
        if llm is not None:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hardware = _runtime_hardware_metadata()
    for metrics in results.values():
        metrics["hardware"] = hardware
        metrics["timing_scope"] = (
            "model-ready through final prediction; model and adapter loading "
            "excluded"
        )

    return results
