import os
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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, OrderedDict
import numpy as np
import random

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
from config import *
from utils import *
from distribution_metrics import compute_distribution_metrics

# ======================================================================
# Evaluation
# ======================================================================

def _score_candidate_logprobs(logits, labels):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    lengths = valid.sum(dim=1).clamp_min(1)
    return token_logps.sum(dim=1), lengths


def _fallback_choice_logprob(wrapper, tokenizer, prompt, valid_options, args,
                             active_loras, use_lora, device):
    """Score legal option numbers only when generation cannot be parsed."""
    if not valid_options:
        return None
    scores = []
    with torch.no_grad():
        for option in sorted(valid_options):
            completion = f" {option}"
            prefix = tokenizer(
                prompt, return_tensors='pt', truncation=True,
                max_length=args.max_length)
            encoded = tokenizer(
                prompt + completion, return_tensors='pt', truncation=True,
                max_length=args.max_length)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = input_ids.clone()
            prefix_len = min(prefix['input_ids'].shape[1], labels.shape[1] - 1)
            labels[:, :prefix_len] = -100
            if use_lora:
                wrapper.set_active_loras(active_loras)
            else:
                wrapper.set_active_loras(set())
            logits = wrapper.base_model(
                input_ids=input_ids, attention_mask=attention_mask).logits
            token_logps, lengths = _score_candidate_logprobs(logits, labels)
            scores.append((float((token_logps / lengths).item()), option))
    return max(scores)[1] if scores else None


def evaluate_choice_logprob(wrapper: MultiLoRAModelWrapper, wvs_data: list,
                            user_indices, question_ids: list,
                            nature_options: dict, tokenizer, args, *,
                            desc: str = "Eval choice logprob",
                            use_lora: bool = True,
                            length_normalize: bool = True) -> dict:
    """Evaluate multiple choice by scoring candidate answer texts.

    This matches the clean SFT training format used by QADataset when
    use_augmented=False: user content is exactly "Question: {qa['question']}"
    and assistant content is the option number plus option text.
    """
    eval_start_time = time.time()
    wrapper.base_model.eval()
    device = wrapper.base_model.get_input_embeddings().weight.device

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    pad_id = (tokenizer.pad_token_id
              if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    has_chat_template = (hasattr(tokenizer, 'apply_chat_template')
                         and tokenizer.chat_template)

    try:
        user_groups: Dict[frozenset, list] = defaultdict(list)
        for idx in sorted(set(user_indices)):
            if idx >= len(wvs_data):
                continue
            try:
                features = get_user_features(
                    wvs_data[idx],
                    dimensions=getattr(args, "feature_dimensions", None),
                    raw_education=getattr(args, "raw_education_features", False),
                )
                if features:
                    user_groups[features].append((idx, wvs_data[idx]))
            except Exception as e:
                print(f"Warning: Error processing user {idx}: {e}")

        correct = 0
        total = 0
        error_sum = 0.0
        sample_outputs_by_features = {}

        with torch.no_grad():
            for features, users in tqdm(sorted(user_groups.items()), desc=desc):
                if use_lora and getattr(args, "joint_adapter_groups", False):
                    active_loras = {joint_adapter_name(features)}
                else:
                    active_loras = features if use_lora else set()

                feature_key = tuple(sorted(features))
                sample_outputs_by_features.setdefault(feature_key, [])

                examples = []
                for _, row in users:
                    for qid in question_ids:
                        q_info = nature_options.get(qid)
                        if not isinstance(q_info, dict):
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
                        opts = q_info.get('options', {})
                        candidate_items = []
                        for key, text in sorted(
                                opts.items(),
                                key=lambda x: (int(x[0])
                                               if x[0].lstrip('-').isdigit()
                                               else 0)):
                            if not key.lstrip('-').isdigit():
                                continue
                            opt_id = int(key)
                            if valid_options is not None and opt_id not in valid_options:
                                continue
                            candidate_items.append(
                                (opt_id, f"{text}\n<answer>{opt_id}</answer>"))
                        if not candidate_items:
                            continue

                        opts_fmt = "\n".join(
                            f"{k}. {v}" for k, v in
                            sorted(opts.items(),
                                   key=lambda x: (int(x[0])
                                                  if x[0].lstrip('-').isdigit()
                                                  else 0))
                        )
                        qa_question = (
                            f"{q_info['question_text']}\nOptions:\n{opts_fmt}")
                        user_content = f"Question: {qa_question}"
                        examples.append({
                            "qid": qid,
                            "gt": gt,
                            "valid_options": valid_options,
                            "user_content": user_content,
                            "candidates": candidate_items,
                        })

                flat = []
                for ex_idx, ex in enumerate(examples):
                    for opt_id, answer_text in ex["candidates"]:
                        if has_chat_template:
                            full_messages = [
                                {"role": "system", "content": ""},
                                {"role": "user", "content": ex["user_content"]},
                                {"role": "assistant", "content": answer_text},
                            ]
                            prefix_messages = [
                                {"role": "system", "content": ""},
                                {"role": "user", "content": ex["user_content"]},
                            ]
                            full = tokenizer.apply_chat_template(
                                full_messages, tokenize=False,
                                add_generation_prompt=False,
                                enable_thinking=False)
                            prefix = tokenizer.apply_chat_template(
                                prefix_messages, tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False)
                        else:
                            prefix = f"{ex['user_content']} "
                            full = prefix + answer_text
                        prefix_ids = tokenizer(
                            prefix, add_special_tokens=False)['input_ids']
                        enc = tokenizer(
                            full, truncation=True, max_length=args.max_length,
                            add_special_tokens=False)
                        ids = enc['input_ids']
                        labels = list(ids)
                        for pos in range(min(len(prefix_ids), len(labels))):
                            labels[pos] = -100
                        if all(x == -100 for x in labels):
                            continue
                        flat.append({
                            "example_index": ex_idx,
                            "option_id": opt_id,
                            "answer_text": answer_text,
                            "input_ids": ids,
                            "labels": labels,
                        })

                candidate_scores = defaultdict(list)
                for start in range(0, len(flat), args.eval_batch_size):
                    batch_items = flat[start:start + args.eval_batch_size]
                    max_len = max(len(x["input_ids"]) for x in batch_items)
                    input_ids = []
                    labels = []
                    attention_mask = []
                    for item in batch_items:
                        pad_n = max_len - len(item["input_ids"])
                        input_ids.append(item["input_ids"] + [pad_id] * pad_n)
                        labels.append(item["labels"] + [-100] * pad_n)
                        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_n)
                    batch = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                        "labels": torch.tensor(labels, dtype=torch.long, device=device),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
                    }
                    if use_lora:
                        wrapper.set_active_loras(active_loras)
                    else:
                        wrapper.set_active_loras(set())
                    out = wrapper.base_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
                    seq_logp, lengths = _score_candidate_logprobs(
                        out.logits, batch["labels"])
                    if length_normalize:
                        scores = seq_logp / lengths
                    else:
                        scores = seq_logp
                    for item, score in zip(batch_items, scores.detach().cpu().tolist()):
                        candidate_scores[item["example_index"]].append({
                            "option_id": item["option_id"],
                            "answer_text": item["answer_text"],
                            "score": float(score),
                        })

                for ex_idx, ex in enumerate(examples):
                    scored = candidate_scores.get(ex_idx, [])
                    if not scored:
                        continue
                    pred = max(scored, key=lambda x: x["score"])["option_id"]
                    if pred == ex["gt"]:
                        correct += 1
                    total += 1
                    if ex["valid_options"]:
                        num_options = len(ex["valid_options"])
                    else:
                        num_options = max(len(ex["candidates"]), 1)
                    error_sum += abs(pred - ex["gt"]) / max(num_options - 1, 1)

                    if len(sample_outputs_by_features[feature_key]) < 5:
                        top_scores = sorted(
                            scored, key=lambda x: x["score"], reverse=True)[:5]
                        sample_outputs_by_features[feature_key].append({
                            "question_id": ex["qid"],
                            "prediction": pred,
                            "ground_truth": ex["gt"],
                            "is_correct": pred == ex["gt"],
                            "top_scores": top_scores,
                            "features": list(features) if use_lora else [],
                        })

    finally:
        tokenizer.padding_side = old_pad_side

    runtime_seconds = time.time() - eval_start_time
    acc = correct / total if total else 0.0
    mae = error_sum / total if total else 0.0
    print(f"  {desc}: {correct}/{total} = {acc:.4f} "
          f"(MAE: {mae:.4f}, time: {runtime_seconds:.1f}s)")

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "invalid_predictions": 0,
        "valid_rate": 1.0 if total else 0.0,
        "mae": mae,
        "valid_count": total,
        "runtime_seconds": runtime_seconds,
        "sample_outputs": {
            "_".join(sorted(features)): samples
            for features, samples in sample_outputs_by_features.items()
        },
        "invalid_outputs": [],
        "scoring": "candidate_answer_text_logprob",
        "length_normalize": length_normalize,
    }

def evaluate(wrapper: MultiLoRAModelWrapper, wvs_data: list,
             user_indices, question_ids: list, nature_options: dict,
             tokenizer, args, *, desc: str = "Eval",
             use_lora: bool = True) -> dict:
    """Evaluate model using generative approach with answer extraction."""
    eval_start_time = time.time()
    wrapper.base_model.eval()
    device = wrapper.base_model.get_input_embeddings().weight.device

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    try:
        user_set = set(user_indices)
        user_groups: Dict[frozenset, list] = defaultdict(list)
        for idx in sorted(user_set):  # Sort for deterministic order
            if idx >= len(wvs_data):
                continue
            try:
                features = get_user_features(
                    wvs_data[idx],
                    dimensions=getattr(args, "feature_dimensions", None),
                    raw_education=getattr(args, "raw_education_features", False),
                )
                if features:
                    user_groups[features].append((idx, wvs_data[idx]))
            except Exception as e:
                print(f"Warning: Error processing user {idx}: {e}")
                continue

        correct = 0
        total = 0
        invalid_predictions = 0
        error_sum = 0  # For normalized MAE calculation
        sample_outputs_by_features = {}  # Store sample outputs by feature combination
        invalid_outputs = []
        distribution_records = []
        input_token_counts = []
        output_token_counts = []

        with torch.no_grad():
            for features, users in tqdm(sorted(user_groups.items()), desc=desc):  # Sort for deterministic order
                if use_lora and getattr(args, "joint_adapter_groups", False):
                    active_loras = {joint_adapter_name(features)}
                else:
                    active_loras = features if use_lora else set()

                # Initialize sample storage for this feature combination
                features_key = tuple(sorted(features))
                if features_key not in sample_outputs_by_features:
                    sample_outputs_by_features[features_key] = []

                prompts, gts, qids, valid_opts_list, participant_ids = [], [], [], [], []
                for participant_id, row in users:
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

                        # Get valid options for this question
                        valid_options = get_valid_option_set(qid, nature_options)
                        if valid_options is not None and gt not in valid_options:
                            continue

                        q_info = nature_options[qid]
                        opts = q_info.get('options', {})
                        q_text = q_info['question_text']
                        opts_fmt = "\n".join(
                            f"{k}. {v}" for k, v in
                            sorted(opts.items(),
                                   key=lambda x: (int(x[0])
                                                  if x[0].lstrip('-').isdigit()
                                                  else 0))
                        )
                        if getattr(args, "train_answer_value_only", False):
                            user_content = (
                                f"Question: {q_text}\n"
                                f"Options:\n{opts_fmt}\n\n"
                                f"Write ONLY your chosen option number.\n\n"
                                f"Answer:")
                        elif getattr(args, "direct_answer_eval", False):
                            user_content = (
                                f"Question: {q_text}\n"
                                f"Options:\n{opts_fmt}\n\n"
                                f"Write ONLY your chosen option number inside "
                                f"<answer></answer> tags.\n\n"
                                f"Required format:\n"
                                f"<answer>[option number]</answer>\n\n"
                                f"Your response:")
                        else:
                            user_content = (f"Question: {q_text}\n"
                                      f"Options:\n{opts_fmt}\n\n"
                                      f"First, provide your reasoning and explanation. "
                                      f"Then, on a new line, write ONLY your chosen option number inside <answer></answer> tags.\n\n"
                                      f"Required format:\n"
                                      f"[Your reasoning and explanation]\n"
                                      f"<answer>[option number]</answer>\n\n"
                                      f"Your response:")

                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                            messages = [
                                {"role": "system", "content": ""},
                                {"role": "user", "content": user_content},
                            ]
                            prompt = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
                        else:
                            prompt = user_content

                        prompts.append(prompt)
                        gts.append(gt)
                        qids.append(qid)
                        valid_opts_list.append(valid_options)
                        participant_ids.append(participant_id)

                if not prompts:
                    continue

                # Process in batches
                for i in range(0, len(prompts), args.eval_batch_size):
                    bp = prompts[i:i + args.eval_batch_size]
                    bg = gts[i:i + args.eval_batch_size]
                    bv = valid_opts_list[i:i + args.eval_batch_size]

                    try:
                        enc = tokenizer(bp, return_tensors='pt', padding=True,
                                        truncation=True, max_length=args.max_length)
                        input_ids = enc['input_ids'].to(device)
                        attn_mask = enc['attention_mask'].to(device)
                        if use_lora:
                            wrapper.set_active_loras(active_loras)
                        else:
                            wrapper.set_active_loras(set())

                        # Generate responses with longer output for reasoning
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

                        # Extract generated text (remove prompt)
                        for j in range(len(bp)):
                            try:
                                prompt_len = input_ids[j].shape[0]
                                generated_ids = gen_ids[j][prompt_len:]
                                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                                input_token_counts.append(int(attn_mask[j].sum().item()))
                                generated_tokens = generated_ids.tolist()
                                eos_ids = tokenizer.eos_token_id
                                if not isinstance(eos_ids, (list, tuple, set)):
                                    eos_ids = [eos_ids] if eos_ids is not None else []
                                output_count = len(generated_tokens)
                                for token_index, token_id in enumerate(generated_tokens):
                                    if token_id in eos_ids:
                                        output_count = token_index + 1
                                        break
                                output_token_counts.append(output_count)

                                # Extract answer number from response
                                prediction = extract_answer_number(response, valid_options=bv[j])
                                if (prediction is None and
                                        getattr(args, 'invalid_prediction_fallback', 'none') == 'choice_logprob'):
                                    prediction = _fallback_choice_logprob(
                                        wrapper, tokenizer, bp[j], bv[j], args,
                                        active_loras, use_lora, device)

                                distribution_records.append({
                                    'participant_id': participant_ids[i + j],
                                    'domain': 'wvs',
                                    'question_id': qids[i + j],
                                    'target': bg[j],
                                    'prediction': prediction,
                                    'valid_options': sorted(bv[j]) if bv[j] else [],
                                })

                                # Store sample outputs (up to 5 per feature combination)
                                if len(sample_outputs_by_features[features_key]) < 5:
                                    sample_outputs_by_features[features_key].append({
                                        'question_id': qids[i + j],
                                        'prompt': bp[j],
                                        'raw_output': response,
                                        'extracted_prediction': prediction,
                                        'ground_truth': bg[j],
                                        'is_correct': prediction == bg[j] if prediction is not None else None,
                                        'features': list(features) if use_lora else []
                                    })

                                if prediction is not None:
                                    abs_error = abs(prediction - bg[j])
                                    # Normalize by (num_options - 1) to get MAE in [0, 1]
                                    num_options = len(bv[j]) if bv[j] else 1
                                    normalized_error = abs_error / max(num_options - 1, 1)
                                    error_sum += normalized_error
                                    if prediction == bg[j]:
                                        correct += 1
                                    total += 1
                                else:
                                    invalid_predictions += 1
                                    total += 1
                                    invalid_outputs.append({
                                        'question_id': qids[i + j],
                                        'prompt': bp[j],
                                        'raw_output': response,
                                        'extracted_prediction': prediction,
                                        'ground_truth': bg[j],
                                        'valid_options': sorted(bv[j]) if bv[j] else None,
                                        'features': list(features) if use_lora else [],
                                    })
                            except Exception as e:
                                print(f"Warning: Error processing response: {e}")
                                invalid_predictions += 1
                                total += 1
                                invalid_outputs.append({
                                    'question_id': qids[i + j] if i + j < len(qids) else None,
                                    'prompt': bp[j] if j < len(bp) else None,
                                    'raw_output': response if 'response' in locals() else "",
                                    'extracted_prediction': None,
                                    'ground_truth': bg[j] if j < len(bg) else None,
                                    'valid_options': sorted(bv[j]) if j < len(bv) and bv[j] else None,
                                    'features': list(features) if use_lora else [],
                                    'error': str(e),
                                })
                                continue

                    except Exception as e:
                        print(f"Warning: Error during batch generation: {e}")
                        # Count all items in this batch as errors
                        invalid_predictions += len(bp)
                        total += len(bp)
                        for j in range(len(bp)):
                            invalid_outputs.append({
                                'question_id': qids[i + j] if i + j < len(qids) else None,
                                'prompt': bp[j],
                                'raw_output': "",
                                'extracted_prediction': None,
                                'ground_truth': bg[j] if j < len(bg) else None,
                                'valid_options': sorted(bv[j]) if j < len(bv) and bv[j] else None,
                                'features': list(features) if use_lora else [],
                                'error': str(e),
                            })
                        continue

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        tokenizer.padding_side = old_pad_side

    acc = correct / total if total > 0 else 0
    valid_rate = (total - invalid_predictions) / total if total > 0 else 0
    valid_count = total - invalid_predictions
    mae = error_sum / valid_count if valid_count > 0 else 0
    runtime_seconds = time.time() - eval_start_time
    print(f"  {desc}: {correct}/{total} = {acc:.4f} "
          f"(MAE: {mae:.4f}, invalid: {invalid_predictions}, "
          f"valid_rate: {valid_rate:.4f}, time: {runtime_seconds:.1f}s)")

    # Convert sample_outputs_by_features to a serializable format
    sample_outputs = {
        '_'.join(sorted(features)): samples
        for features, samples in sample_outputs_by_features.items()
    }

    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'invalid_predictions': invalid_predictions,
        'valid_rate': valid_rate,
        'mae': mae,
        'valid_count': valid_count,
        'runtime_seconds': runtime_seconds,
        'sample_outputs': sample_outputs,  # Include sample outputs by feature combination
        'prediction_records': distribution_records,
        'invalid_outputs': invalid_outputs,
        'token_usage': {
            'input_tokens': sum(input_token_counts),
            'output_tokens': sum(output_token_counts),
            'total_tokens': sum(input_token_counts) + sum(output_token_counts),
        },
        **compute_distribution_metrics(distribution_records),
    }


def evaluate_with_persona_prompt(wrapper: MultiLoRAModelWrapper, wvs_data: list,
                                  user_indices, question_ids: list, nature_options: dict,
                                  tokenizer, args, *, desc: str = "Eval with Persona",
                                  use_lora: bool = False) -> dict:
    """Evaluate model with persona prompt prepended to each question.

    This function tests the effect of adding demographic information as a text prompt
    (e.g., "I am a male survey respondent. I am from Asia. I am a young person.")
    before each question, without using LoRA adapters.
    """
    eval_start_time = time.time()
    wrapper.base_model.eval()
    device = wrapper.base_model.get_input_embeddings().weight.device

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    try:
        user_set = set(user_indices)
        user_groups: Dict[frozenset, list] = defaultdict(list)
        for idx in sorted(user_set):  # Sort for deterministic order
            if idx >= len(wvs_data):
                continue
            try:
                features = get_user_features(
                    wvs_data[idx],
                    dimensions=getattr(args, "feature_dimensions", None),
                    raw_education=getattr(args, "raw_education_features", False),
                )
                if features:
                    user_groups[features].append((idx, wvs_data[idx]))
            except Exception as e:
                print(f"Warning: Error processing user {idx}: {e}")
                continue

        correct = 0
        total = 0
        invalid_predictions = 0
        error_sum = 0
        sample_outputs_by_features = {}  # Store sample outputs by feature combination
        invalid_outputs = []

        with torch.no_grad():
            for features, users in tqdm(sorted(user_groups.items()), desc=desc):  # Sort for deterministic order
                # Generate persona prompt for this feature group
                profile_str = generate_persona_prompt(features)

                # Set LoRA adapters (usually disabled for this test)
                if use_lora and getattr(args, "joint_adapter_groups", False):
                    active_loras = {joint_adapter_name(features)}
                else:
                    active_loras = features if use_lora else set()

                # Initialize sample storage for this feature combination
                features_key = tuple(sorted(features))
                if features_key not in sample_outputs_by_features:
                    sample_outputs_by_features[features_key] = []

                prompts, gts, qids, valid_opts_list = [], [], [], []
                for _, row in users:
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

                        # Get valid options for this question
                        valid_options = get_valid_option_set(qid, nature_options)
                        if valid_options is not None and gt not in valid_options:
                            continue

                        q_info = nature_options[qid]
                        opts = q_info.get('options', {})
                        q_text = q_info['question_text']
                        opts_fmt = "\n".join(
                            f"{k}. {v}" for k, v in
                            sorted(opts.items(),
                                   key=lambda x: (int(x[0])
                                                  if x[0].lstrip('-').isdigit()
                                                  else 0))
                        )
                        # Build messages with system and user roles
                        sys_prompt = f"You are a person with the following profile: {profile_str}. You are a helpful assistant that answers survey questions honestly."
                        if getattr(args, "train_answer_value_only", False):
                            user_prompt = (
                                f"Question: {q_text}\n"
                                f"Options:\n{opts_fmt}\n\n"
                                f"Write ONLY your chosen option number.\n\n"
                                f"Answer:")
                        elif getattr(args, "direct_answer_eval", False):
                            user_prompt = (
                                f"Question: {q_text}\n"
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

                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}
                        ]

                        # Apply chat template if available
                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        else:
                            # Fallback: concatenate system and user messages
                            prompt = f"{sys_prompt}\n\n{user_prompt}"

                        prompts.append(prompt)
                        gts.append(gt)
                        qids.append(qid)
                        valid_opts_list.append(valid_options)

                if not prompts:
                    continue

                # Process in batches
                for i in range(0, len(prompts), args.eval_batch_size):
                    bp = prompts[i:i + args.eval_batch_size]
                    bg = gts[i:i + args.eval_batch_size]
                    bv = valid_opts_list[i:i + args.eval_batch_size]

                    try:
                        enc = tokenizer(bp, return_tensors='pt', padding=True,
                                        truncation=True, max_length=args.max_length)
                        input_ids = enc['input_ids'].to(device)
                        attn_mask = enc['attention_mask'].to(device)
                        if use_lora:
                            wrapper.set_active_loras(active_loras)
                        else:
                            wrapper.set_active_loras(set())

                        # Generate responses
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

                        # Extract generated text
                        for j in range(len(bp)):
                            try:
                                prompt_len = input_ids[j].shape[0]
                                generated_ids = gen_ids[j][prompt_len:]
                                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                                # Extract answer number from response
                                prediction = extract_answer_number(response, valid_options=bv[j])

                                # Store sample outputs (up to 5 per feature combination)
                                if len(sample_outputs_by_features[features_key]) < 5:
                                    sample_outputs_by_features[features_key].append({
                                        'question_id': qids[i + j],
                                        'prompt': bp[j],
                                        'raw_output': response,
                                        'extracted_prediction': prediction,
                                        'ground_truth': bg[j],
                                        'is_correct': prediction == bg[j] if prediction is not None else None,
                                        'features': list(features) if use_lora else [],
                                        'persona_prompt': profile_str
                                    })

                                if prediction is not None:
                                    abs_error = abs(prediction - bg[j])
                                    num_options = len(bv[j]) if bv[j] else 1
                                    normalized_error = abs_error / max(num_options - 1, 1)
                                    error_sum += normalized_error
                                    if prediction == bg[j]:
                                        correct += 1
                                    total += 1
                                else:
                                    invalid_predictions += 1
                                    total += 1
                                    invalid_outputs.append({
                                        'question_id': qids[i + j],
                                        'prompt': bp[j],
                                        'raw_output': response,
                                        'extracted_prediction': prediction,
                                        'ground_truth': bg[j],
                                        'valid_options': sorted(bv[j]) if bv[j] else None,
                                        'features': list(features) if use_lora else [],
                                        'persona_prompt': profile_str,
                                    })
                            except Exception as e:
                                print(f"Warning: Error processing response: {e}")
                                invalid_predictions += 1
                                total += 1
                                invalid_outputs.append({
                                    'question_id': qids[i + j] if i + j < len(qids) else None,
                                    'prompt': bp[j] if j < len(bp) else None,
                                    'raw_output': response if 'response' in locals() else "",
                                    'extracted_prediction': None,
                                    'ground_truth': bg[j] if j < len(bg) else None,
                                    'valid_options': sorted(bv[j]) if j < len(bv) and bv[j] else None,
                                    'features': list(features) if use_lora else [],
                                    'persona_prompt': profile_str,
                                    'error': str(e),
                                })
                                continue

                    except Exception as e:
                        print(f"Warning: Error during batch generation: {e}")
                        invalid_predictions += len(bp)
                        total += len(bp)
                        for j in range(len(bp)):
                            invalid_outputs.append({
                                'question_id': qids[i + j] if i + j < len(qids) else None,
                                'prompt': bp[j],
                                'raw_output': "",
                                'extracted_prediction': None,
                                'ground_truth': bg[j] if j < len(bg) else None,
                                'valid_options': sorted(bv[j]) if j < len(bv) and bv[j] else None,
                                'features': list(features) if use_lora else [],
                                'persona_prompt': profile_str,
                                'error': str(e),
                            })
                        continue

    except Exception as e:
        print(f"Error during evaluation with persona prompt: {e}")
        raise
    finally:
        tokenizer.padding_side = old_pad_side

    acc = correct / total if total > 0 else 0
    valid_rate = (total - invalid_predictions) / total if total > 0 else 0
    valid_count = total - invalid_predictions
    mae = error_sum / valid_count if valid_count > 0 else 0
    runtime_seconds = time.time() - eval_start_time
    print(f"  {desc}: {correct}/{total} = {acc:.4f} "
          f"(MAE: {mae:.4f}, invalid: {invalid_predictions}, "
          f"valid_rate: {valid_rate:.4f}, time: {runtime_seconds:.1f}s)")

    # Convert sample_outputs_by_features to a serializable format
    sample_outputs = {
        '_'.join(sorted(features)): samples
        for features, samples in sample_outputs_by_features.items()
    }

    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'invalid_predictions': invalid_predictions,
        'valid_rate': valid_rate,
        'mae': mae,
        'valid_count': valid_count,
        'runtime_seconds': runtime_seconds,
        'sample_outputs': sample_outputs,  # Include sample outputs by feature combination
        'invalid_outputs': invalid_outputs,
    }
