"""PARADE's composable Task/Demographic LoRA implementation.

Main-experiment setting
-----------------------
The backbone is frozen and LoRA modules are attached to the attention
projections (q/k/v/o) and MLP projections (gate/up/down).  The experiment
scripts use eight demographic dimensions:

    gender, age group, country, religion, education, marital status,
    employment status, and urban/rural residence.

Adapter names are discovered from the data rather than fixed here.  In the
locked WVS setting this gives 106 Demographic LoRAs (2 + 3 + 66 + 10 + 9 +
6 + 8 + 2 attribute values) plus one shared ``task_shared`` LoRA.  SocioBench
uses the same eight-dimensional schema, with the value set observed in that
dataset.

Two-stage training and composition
----------------------------------
Stage 1 trains ``task_shared`` on demographic-free survey examples with
balanced, randomly sampled legal option identifiers in every main run, so
that it captures the common multiple-choice response protocol.  Stage 2
freezes both the backbone and Task LoRA, activates one Demographic LoRA per
available dimension for each respondent, and jointly updates those eight
modules from the same answer-token loss.  A soft penalty on
``A_task @ A_demographic.T`` discourages task/demographic row-space overlap.

The main setting also learns one bounded multiplier per demographic dimension,
``w_d = 1 + 0.5 * tanh(eta_d)``, initialized at one.  The effective update is
the shared Task LoRA plus the weighted sum of active Demographic LoRAs.

Inference and export
--------------------
Evaluation activates the Task LoRA together with the eight Demographic LoRAs
selected by a respondent profile.  Per-attribute adapters are saved in
PEFT-compatible form.  For vLLM, the active Task and Demographic adapters can
be concatenated into one higher-rank PEFT adapter whose matrix product equals
their weighted sum.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Set
import numpy as np
import random

from anonymization import public_path

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

# Library fallback.  Main experiment scripts override this with q/k/v/o plus
# gate/up/down projections; keeping the fallback small preserves legacy use.
TARGET_MODULES = ['gate_proj', 'up_proj', 'down_proj']

# ======================================================================
# Multi-LoRA Modules
# ======================================================================

class MultiLoRALinear(nn.Module):
    """Linear layer with selectable, optionally weighted LoRA contributions."""

    def __init__(self, base_linear: nn.Linear, lora_names: List[str],
                 rank: int, alpha: float, dropout: float = 0.0,
                 composition_mode: str = "sum",
                 orthogonal_eps: float = 1e-6,
                 orthogonal_strength: float = 1.0,
                 task_scale: float = 1.0,
                 knowledge_scale: float = 1.0,
                 task_lora_name: str = "task_shared"):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.composition_mode = composition_mode
        self.orthogonal_eps = max(float(orthogonal_eps), 1e-12)
        self.orthogonal_strength = float(orthogonal_strength)
        self.task_scale = float(task_scale)
        self.knowledge_scale = float(knowledge_scale)
        self.task_lora_name = str(task_lora_name)

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in lora_names:
            A = nn.Parameter(torch.empty(rank, self.in_features,
                                         dtype=dtype, device=device))
            B = nn.Parameter(torch.zeros(self.out_features, rank,
                                         dtype=dtype, device=device))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A[name] = A
            self.lora_B[name] = B

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._active: List[str] = []
        self._active_weights: Optional[torch.Tensor] = None

    def set_active_loras(self, names: Sequence[str],
                         weights: Optional[torch.Tensor] = None):
        self._active = [n for n in names if n in self.lora_A] if names else []
        self._active_weights = weights

    def _low_rank_inner(self, left: str, right: str, *, detach: bool = False):
        a_left = self.lora_A[left].float()
        a_right = self.lora_A[right].float()
        b_left = self.lora_B[left].float()
        b_right = self.lora_B[right].float()
        if detach:
            a_left = a_left.detach()
            a_right = a_right.detach()
            b_left = b_left.detach()
            b_right = b_right.detach()
        a_dot = a_left.matmul(a_right.t())
        b_dot = b_left.t().matmul(b_right)
        return (a_dot * b_dot).sum()

    def _adapter_gram(self, names: List[str], *, detach: bool = False):
        rows = []
        for left in names:
            rows.append(torch.stack([
                self._low_rank_inner(left, right, detach=detach)
                for right in names
            ]))
        return torch.stack(rows)

    def _orthogonal_basis_matrix(self, names: List[str],
                                 *, detach: bool = False):
        """Rows are effective orthogonalized adapters in raw-adapter basis."""
        if len(names) < 2:
            eye = torch.eye(
                len(names),
                device=self.lora_A[names[0]].device if names else None,
                dtype=torch.float32,
            )
            return eye

        gram = self._adapter_gram(names, detach=detach)
        eye = torch.eye(len(names), device=gram.device, dtype=gram.dtype)
        basis = []
        for idx in range(len(names)):
            coeff = eye[idx].clone()
            for prev in basis:
                denom = prev.matmul(gram).dot(prev)
                if float(denom.detach().abs().cpu()) <= self.orthogonal_eps:
                    continue
                numer = gram[idx].dot(prev)
                projection = (
                    self.orthogonal_strength
                    * numer
                    / denom.clamp_min(self.orthogonal_eps)
                )
                coeff = coeff - projection * prev
            basis.append(coeff)
        return torch.stack(basis)

    def _effective_coefficients(self, names: List[str],
                                weights: Optional[torch.Tensor]):
        if len(names) < 2:
            return weights
        if self.composition_mode == "orthogonal_projection":
            basis = self._orthogonal_basis_matrix(names, detach=False)
            if weights is None:
                weights = torch.ones(len(names), device=basis.device,
                                     dtype=basis.dtype)
            else:
                weights = weights.to(device=basis.device, dtype=basis.dtype)
            if weights.dim() == 1:
                return weights.matmul(basis)
            return weights.matmul(basis)
        if self.composition_mode != "task_knowledge_projection":
            if self.composition_mode == "explicit_task_knowledge_projection":
                return self._effective_explicit_task_knowledge_coefficients(
                    names, weights)
            return weights

        basis = self._task_knowledge_basis_matrix(names, detach=False)
        if weights is None:
            weights = torch.ones(len(names), device=basis.device,
                                 dtype=basis.dtype)
        else:
            weights = weights.to(device=basis.device, dtype=basis.dtype)
        if weights.dim() == 1:
            task_coeff = basis[0] * self.task_scale
            knowledge_coeffs = weights.matmul(basis[1:]) * self.knowledge_scale
            return task_coeff + knowledge_coeffs
        task_coeff = basis[0].unsqueeze(0) * self.task_scale
        knowledge_coeffs = weights.matmul(basis[1:]) * self.knowledge_scale
        return task_coeff + knowledge_coeffs

    def _effective_explicit_task_knowledge_coefficients(
            self, names: List[str], weights: Optional[torch.Tensor]):
        if self.task_lora_name not in names:
            return weights
        if len(names) < 2:
            return weights

        basis, knowledge_names = self._explicit_task_knowledge_basis_matrix(
            names, detach=False)
        if not knowledge_names:
            return basis[0] * self.task_scale
        if weights is None:
            weights = torch.ones(len(knowledge_names), device=basis.device,
                                 dtype=basis.dtype)
        else:
            weights = weights.to(device=basis.device, dtype=basis.dtype)
            if weights.numel() == len(names):
                keep = [idx for idx, name in enumerate(names)
                        if name != self.task_lora_name]
                weights = weights.index_select(
                    0, torch.tensor(keep, device=basis.device))
        if weights.dim() == 1:
            task_coeff = basis[0] * self.task_scale
            knowledge_coeffs = weights.matmul(basis[1:]) * self.knowledge_scale
            return task_coeff + knowledge_coeffs
        task_coeff = basis[0].unsqueeze(0) * self.task_scale
        knowledge_coeffs = weights.matmul(basis[1:]) * self.knowledge_scale
        return task_coeff + knowledge_coeffs

    def _task_knowledge_basis_matrix(self, names: List[str],
                                     *, detach: bool = False):
        """First row is the shared task direction; remaining rows are
        per-adapter residual knowledge directions orthogonalized against it.
        Rows are expressed in the raw-adapter basis."""
        if len(names) < 2:
            return torch.eye(
                len(names),
                device=self.lora_A[names[0]].device if names else None,
                dtype=torch.float32,
            )

        n = len(names)
        device = self.lora_A[names[0]].device
        eye = torch.eye(n, device=device, dtype=torch.float32)
        task = torch.full(
            (n,),
            1.0 / float(n),
            device=device,
            dtype=torch.float32,
        )
        if self.orthogonal_strength == 0.0:
            return torch.cat((task.unsqueeze(0), eye), dim=0)

        gram = self._adapter_gram(names, detach=detach)
        task = task.to(device=gram.device, dtype=gram.dtype)
        task_denom = task.matmul(gram).dot(task)
        eye = eye.to(device=gram.device, dtype=gram.dtype)
        rows = [task]
        for idx in range(n):
            coeff = eye[idx].clone()
            if float(task_denom.detach().abs().cpu()) > self.orthogonal_eps:
                numer = gram[idx].dot(task)
                projection = (
                    self.orthogonal_strength
                    * numer
                    / task_denom.clamp_min(self.orthogonal_eps)
                )
                coeff = coeff - projection * task
            rows.append(coeff)
        return torch.stack(rows)

    def _explicit_task_knowledge_basis_matrix(self, names: List[str],
                                              *, detach: bool = False):
        """Use a separately trained task adapter as the task direction.

        First row is the explicit task adapter. Remaining rows are demographic
        adapters with their component along the task adapter removed.
        """
        if self.task_lora_name not in names:
            return self._task_knowledge_basis_matrix(names, detach=detach), names

        n = len(names)
        task_idx = names.index(self.task_lora_name)
        device = self.lora_A[names[0]].device
        eye = torch.eye(n, device=device, dtype=torch.float32)
        task = eye[task_idx].clone()
        knowledge_indices = [
            idx for idx, name in enumerate(names)
            if name != self.task_lora_name
        ]
        knowledge_names = [names[idx] for idx in knowledge_indices]
        if self.orthogonal_strength == 0.0:
            rows = [task] + [eye[idx].clone() for idx in knowledge_indices]
            return torch.stack(rows), knowledge_names

        task_name = names[task_idx]
        task_denom = self._low_rank_inner(
            task_name, task_name, detach=detach)
        eye = eye.to(device=task_denom.device, dtype=task_denom.dtype)
        task = task.to(device=task_denom.device, dtype=task_denom.dtype)
        has_task_direction = task_denom.abs() > self.orthogonal_eps
        safe_denom = torch.where(
            has_task_direction, task_denom, torch.ones_like(task_denom))
        rows = [task]
        for idx, name in enumerate(names):
            if name == self.task_lora_name:
                continue
            coeff = eye[idx].clone()
            numer = self._low_rank_inner(name, task_name, detach=detach)
            projection = torch.where(
                has_task_direction,
                self.orthogonal_strength * numer / safe_denom,
                torch.zeros_like(numer),
            )
            coeff = coeff - projection * task
            rows.append(coeff)
        return torch.stack(rows), knowledge_names

    def merge_coefficients(self, names: List[str],
                           weights: Optional[Dict[str, float]] = None):
        names = [n for n in names if n in self.lora_A]
        if not names:
            return {}
        weight_vec = None
        if weights is not None:
            weight_vec = torch.tensor(
                [float(weights.get(n, 1.0)) for n in names],
                device=self.lora_A[names[0]].device,
                dtype=torch.float32,
            )
        if self.composition_mode == "orthogonal_projection" and len(names) > 1:
            basis = self._orthogonal_basis_matrix(names, detach=True)
            if weight_vec is None:
                weight_vec = torch.ones(len(names), device=basis.device,
                                        dtype=basis.dtype)
            coeffs = weight_vec.matmul(basis)
        elif (self.composition_mode == "task_knowledge_projection"
              and len(names) > 1):
            basis = self._task_knowledge_basis_matrix(names, detach=True)
            if weight_vec is None:
                weight_vec = torch.ones(len(names), device=basis.device,
                                        dtype=basis.dtype)
            coeffs = (
                basis[0] * self.task_scale
                + weight_vec.matmul(basis[1:]) * self.knowledge_scale
            )
        elif (self.composition_mode == "explicit_task_knowledge_projection"
              and len(names) > 1
              and self.task_lora_name in names):
            basis, knowledge_names = self._explicit_task_knowledge_basis_matrix(
                names, detach=True)
            if knowledge_names:
                if weight_vec is None:
                    knowledge_weights = torch.ones(
                        len(knowledge_names), device=basis.device,
                        dtype=basis.dtype)
                else:
                    keep = [idx for idx, name in enumerate(names)
                            if name != self.task_lora_name]
                    knowledge_weights = weight_vec.index_select(
                        0, torch.tensor(keep, device=basis.device))
                coeffs = (
                    basis[0] * self.task_scale
                    + knowledge_weights.matmul(basis[1:]) * self.knowledge_scale
                )
            else:
                coeffs = basis[0] * self.task_scale
        else:
            if weight_vec is None:
                coeffs = torch.ones(len(names), device=self.lora_A[names[0]].device,
                                    dtype=torch.float32)
            else:
                coeffs = weight_vec
        return {
            name: float(coeff.detach().cpu())
            for name, coeff in zip(names, coeffs)
        }

    def forward(self, x):
        result = self.base_linear(x)
        if not self._active:
            return result

        x_d = self.lora_dropout(x)
        lora_out = None
        coeffs = self._effective_coefficients(self._active, self._active_weights)
        if coeffs is not None:
            coeffs = coeffs.to(device=x.device, dtype=x.dtype)

        for pos, name in enumerate(self._active):
            contrib = F.linear(F.linear(x_d, self.lora_A[name]),
                               self.lora_B[name])
            if coeffs is not None:
                if coeffs.dim() == 1:
                    contrib = contrib * coeffs[pos]
                else:
                    view_shape = [coeffs.size(0)] + [1] * (contrib.dim() - 1)
                    contrib = contrib * coeffs[:, pos].view(*view_shape)
            lora_out = contrib if lora_out is None else (lora_out + contrib)

        if lora_out is not None:
            result = result + lora_out * self.scaling
        return result


class MultiLoRAModelWrapper:
    """Manages injection and lifecycle of multiple LoRA adapters on a
    pretrained causal-LM."""

    def __init__(self, base_model, lora_names: List[str], *,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.05,
                 model_name: str = "", weight_mode: str = "uniform",
                 weight_normalize: str = "sum_to_active_count",
                 composition_mode: str = "sum",
                 dimension_order: Optional[List[str]] = None,
                 orthogonal_eps: float = 1e-6,
                 orthogonal_strength: float = 1.0,
                 task_scale: float = 1.0,
                 knowledge_scale: float = 1.0,
                 task_lora_name: str = "task_shared",
                 target_modules: Optional[Sequence[str]] = None):
        self.base_model = base_model
        self.lora_names = list(lora_names)
        self.lora_name_to_idx = {name: i for i, name in enumerate(self.lora_names)}
        self.rank = rank
        self.alpha = alpha
        self.model_name = model_name
        self.weight_mode = weight_mode
        self.weight_normalize = weight_normalize
        self.composition_mode = composition_mode
        self.dimension_order = list(dimension_order or [])
        self.dimension_order_index = {
            dim: i for i, dim in enumerate(self.dimension_order)
        }
        self.orthogonal_eps = orthogonal_eps
        self.orthogonal_strength = orthogonal_strength
        self.task_scale = float(task_scale)
        self.knowledge_scale = float(knowledge_scale)
        self.task_lora_name = str(task_lora_name)
        self.target_modules = list(dict.fromkeys(
            str(name).strip() for name in (target_modules or TARGET_MODULES)
            if str(name).strip()
        ))
        if not self.target_modules:
            raise ValueError("target_modules must contain at least one name")
        self.target_module_prefixes = self._target_prefixes(base_model)
        self.lora_layers: List[MultiLoRALinear] = []
        self.layer_paths: List[str] = []
        self.static_weight_logits = None
        self.gate_dimensions: List[str] = []
        self.lora_name_to_gate_dimension: Dict[str, str] = {}
        self.gate_dimension_to_idx: Dict[str, int] = {}

        self._inject(dropout)
        self._init_weighting()
        self._freeze_and_unfreeze()

        if hasattr(base_model, 'enable_input_require_grads'):
            base_model.enable_input_require_grads()

    # ----- injection -----
    @staticmethod
    def _target_prefixes(base_model) -> tuple:
        config = getattr(base_model, "config", None)
        model_type = str(getattr(config, "model_type", "") or "")
        if model_type == "mistral3":
            if hasattr(base_model, "language_model"):
                return ("language_model.",)
            inner = getattr(base_model, "model", None)
            if inner is not None and hasattr(inner, "language_model"):
                return ("model.language_model.",)
        return ()

    def _inject(self, dropout: float):
        replaced = 0
        for mod_name, mod in self.base_model.named_modules():
            if (self.target_module_prefixes
                    and not mod_name.startswith(self.target_module_prefixes)):
                continue
            for tgt in self.target_modules:
                if not hasattr(mod, tgt):
                    continue
                old = getattr(mod, tgt)
                if not isinstance(old, nn.Linear):
                    continue
                new_layer = MultiLoRALinear(
                    old, self.lora_names, self.rank, self.alpha, dropout,
                    composition_mode=self.composition_mode,
                    orthogonal_eps=self.orthogonal_eps,
                    orthogonal_strength=self.orthogonal_strength,
                    task_scale=self.task_scale,
                    knowledge_scale=self.knowledge_scale,
                    task_lora_name=self.task_lora_name,
                )
                setattr(mod, tgt, new_layer)
                self.lora_layers.append(new_layer)
                self.layer_paths.append(f"{mod_name}.{tgt}")
                replaced += 1
        if not replaced:
            raise ValueError(
                "No nn.Linear layers matched target_modules="
                f"{self.target_modules}")
        print(f"Injected multi-LoRA into {replaced} layers "
              f"({len(self.lora_names)} adapters each; "
              f"targets={self.target_modules})")

    def _init_weighting(self):
        if self.weight_mode in (None, "uniform"):
            self.weight_mode = "uniform"
            return
        if self.weight_mode == "static":
            emb = self.base_model.get_input_embeddings().weight
            self.static_weight_logits = nn.Parameter(
                torch.zeros(len(self.lora_names), device=emb.device,
                            dtype=torch.float32))
            return
        if self.weight_mode == "residual_gate":
            emb = self.base_model.get_input_embeddings().weight
            self.lora_name_to_gate_dimension = {
                name: self._feature_dimension(name) for name in self.lora_names
            }
            self.gate_dimensions = sorted(set(
                self.lora_name_to_gate_dimension.values()))
            self.gate_dimension_to_idx = {
                dim: i for i, dim in enumerate(self.gate_dimensions)
            }
            self.static_weight_logits = nn.Parameter(
                torch.zeros(len(self.gate_dimensions), device=emb.device,
                            dtype=torch.float32))
            print(f"Residual-gate dimensions ({len(self.gate_dimensions)}): "
                  f"{self.gate_dimensions}")
            return
        raise ValueError(f"Unknown LoRA weight mode: {self.weight_mode}")

    @staticmethod
    def _feature_dimension(name: str) -> str:
        if name == "task_shared" or name.startswith("task_"):
            return "task"
        if name in ("male", "female"):
            return "gender"
        if name.startswith("age_") or name in ("young", "middle_aged", "old"):
            return "age_group"
        if name.startswith("country_"):
            return "country"
        if name.startswith("edu_"):
            return "education"
        if name.startswith("marital_"):
            return "marital_status"
        if name.startswith("rel_"):
            return "religion"
        if name.startswith("eth_"):
            return "ethnicity"
        if name.startswith("employment_"):
            return "employment"
        if name in ("urban", "rural"):
            return "urban_rural"
        if name.startswith("occupation_"):
            return "occupation"
        if name in {
            "asia", "europe", "north_america", "south_america",
            "africa", "oceania",
        }:
            return "region"
        return name

    def _freeze_and_unfreeze(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        for layer in self.lora_layers:
            for p in layer.lora_A.values():
                p.requires_grad_(True)
            for p in layer.lora_B.values():
                p.requires_grad_(True)
        if self.static_weight_logits is not None:
            self.static_weight_logits.requires_grad_(True)
        trainable = sum(p.numel() for p in self.base_model.parameters()
                        if p.requires_grad)
        if self.static_weight_logits is not None:
            trainable += self.static_weight_logits.numel()
        total = sum(p.numel() for p in self.base_model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} "
              f"({trainable / total * 100:.4f}%)")

    def set_trainable_feature_dimensions(self, dimensions: Sequence[str]):
        dims = set(dimensions or [])
        for layer in self.lora_layers:
            for name in self.lora_names:
                trainable = self._feature_dimension(name) in dims
                if name in layer.lora_A:
                    layer.lora_A[name].requires_grad_(trainable)
                if name in layer.lora_B:
                    layer.lora_B[name].requires_grad_(trainable)
        trainable_adapters = [
            name for name in self.lora_names
            if self._feature_dimension(name) in dims
        ]
        print("Trainable feature dimensions: "
              f"{sorted(dims)} ({len(trainable_adapters)} adapters)")
        self._print_trainable_count()

    def set_trainable_lora_names(self, trainable_names: Sequence[str],
                                 *, train_weighting: bool = True):
        names = set(trainable_names or [])
        for layer in self.lora_layers:
            for name in self.lora_names:
                trainable = name in names
                if name in layer.lora_A:
                    layer.lora_A[name].requires_grad_(trainable)
                if name in layer.lora_B:
                    layer.lora_B[name].requires_grad_(trainable)
        if self.static_weight_logits is not None:
            self.static_weight_logits.requires_grad_(bool(train_weighting))
        print("Trainable LoRA names: "
              f"{sorted(names)} ({len(names)} adapters); "
              f"train_weighting={bool(train_weighting)}")
        self._print_trainable_count()

    def _print_trainable_count(self):
        trainable = sum(p.numel() for p in self.base_model.parameters()
                        if p.requires_grad)
        if self.static_weight_logits is not None and self.static_weight_logits.requires_grad:
            trainable += self.static_weight_logits.numel()
        total = sum(p.numel() for p in self.base_model.parameters())
        print(f"Trainable after filtering: {trainable:,} / {total:,} "
              f"({trainable / total * 100:.4f}%)")

    # ----- runtime -----
    def _static_weights(self, active_names: List[str]):
        if self.static_weight_logits is None or not active_names:
            return None
        idx = torch.tensor(
            [self.lora_name_to_idx[n] for n in active_names],
            device=self.static_weight_logits.device,
            dtype=torch.long,
        )
        logits = self.static_weight_logits.index_select(0, idx)
        weights = torch.softmax(logits, dim=-1)
        if self.weight_normalize == "sum_to_active_count":
            weights = weights * float(len(active_names))
        elif self.weight_normalize == "none":
            weights = torch.sigmoid(logits)
        return weights

    def _residual_gate_weights(self, active_names: List[str]):
        """Return dimension-shared multipliers in [0.5, 1.5], initialized at 1."""
        if self.static_weight_logits is None or not active_names:
            return None
        idx = torch.tensor(
            [
                self.gate_dimension_to_idx[
                    self.lora_name_to_gate_dimension[n]
                ]
                for n in active_names
            ],
            device=self.static_weight_logits.device,
            dtype=torch.long,
        )
        logits = self.static_weight_logits.index_select(0, idx)
        return 1.0 + 0.5 * torch.tanh(logits)

    def _order_lora_names(self, names) -> List[str]:
        active = [n for n in set(names) if n in self.lora_name_to_idx]
        return sorted(
            active,
            key=lambda n: (
                self.dimension_order_index.get(
                    self._feature_dimension(n),
                    len(self.dimension_order_index),
                ),
                self._feature_dimension(n),
                n,
            ),
        )

    def compute_lora_weights(self, names):
        active_names = self._order_lora_names(names) if names else []
        if (self.composition_mode == "explicit_task_knowledge_projection"
                and self.task_lora_name in active_names):
            active_names = [
                name for name in active_names
                if name != self.task_lora_name
            ]
        if not active_names or self.weight_mode == "uniform":
            return None
        if self.weight_mode == "static":
            return self._static_weights(active_names)
        if self.weight_mode == "residual_gate":
            return self._residual_gate_weights(active_names)
        return None

    def set_active_loras(self, names, weights: Optional[torch.Tensor] = None):
        active_names = self._order_lora_names(names) if names else []
        if (self.composition_mode == "explicit_task_knowledge_projection"
                and active_names
                and self.task_lora_name in self.lora_name_to_idx
                and self.task_lora_name not in active_names):
            active_names = self._order_lora_names(
                list(active_names) + [self.task_lora_name])
        if weights is None and self.weight_mode == "static":
            weights = self._static_weights(active_names)
        if weights is None and self.weight_mode == "residual_gate":
            weight_names = active_names
            if self.composition_mode == "explicit_task_knowledge_projection":
                weight_names = [
                    name for name in active_names
                    if name != self.task_lora_name
                ]
            weights = self._residual_gate_weights(weight_names)
        for layer in self.lora_layers:
            layer.set_active_loras(active_names, weights)

    def set_active_loras_exact(self, names,
                               weights: Optional[torch.Tensor] = None):
        """Activate exactly the requested adapters without implicit task LoRA."""
        active_names = self._order_lora_names(names) if names else []
        if weights is None and self.weight_mode == "static":
            weights = self._static_weights(active_names)
        if weights is None and self.weight_mode == "residual_gate":
            weight_names = active_names
            if self.composition_mode == "explicit_task_knowledge_projection":
                weight_names = [
                    name for name in active_names
                    if name != self.task_lora_name
                ]
            weights = self._residual_gate_weights(weight_names)
        for layer in self.lora_layers:
            layer.set_active_loras(active_names, weights)

    def set_composition_scales(self, *, task_scale: Optional[float] = None,
                               knowledge_scale: Optional[float] = None):
        """Update inference-time task/knowledge scales on every LoRA layer."""
        if task_scale is not None:
            self.task_scale = float(task_scale)
        if knowledge_scale is not None:
            self.knowledge_scale = float(knowledge_scale)
        for layer in self.lora_layers:
            layer.task_scale = self.task_scale
            layer.knowledge_scale = self.knowledge_scale

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for layer in self.lora_layers:
            params.extend(p for p in layer.lora_A.values() if p.requires_grad)
            params.extend(p for p in layer.lora_B.values() if p.requires_grad)
        if (self.static_weight_logits is not None
                and self.static_weight_logits.requires_grad):
            params.append(self.static_weight_logits)
        return params

    def weighting_parameters(self) -> List[nn.Parameter]:
        params = []
        if (self.static_weight_logits is not None
                and self.static_weight_logits.requires_grad):
            params.append(self.static_weight_logits)
        return params

    def soft_orthogonality_loss(self, task_lora_name: str,
                                knowledge_names: Sequence[str],
                                *, normalize: bool = True):
        """Penalty ||A_task A_knowledge^T||_F^2 over adapted layers.

        This matches the soft task/knowledge row-space orthogonality objective
        used when training knowledge LoRAs on top of a frozen task LoRA.
        """
        task = str(task_lora_name)
        names = [n for n in knowledge_names if n != task]
        if not names:
            return None

        total = None
        count = 0
        for layer in self.lora_layers:
            if task not in layer.lora_A:
                continue
            a_task = layer.lora_A[task].float()
            for name in names:
                if name not in layer.lora_A:
                    continue
                a_knowledge = layer.lora_A[name].float()
                cross = a_task.matmul(a_knowledge.t())
                penalty = cross.pow(2).sum()
                if normalize:
                    penalty = penalty / max(cross.numel(), 1)
                total = penalty if total is None else total + penalty
                count += 1

        if total is None:
            return None
        if normalize and count > 0:
            total = total / float(count)
        return total

    def lora_parameters(self, lora_name: str) -> List[nn.Parameter]:
        params = []
        for layer in self.lora_layers:
            if (lora_name in layer.lora_A
                    and layer.lora_A[lora_name].requires_grad):
                params.append(layer.lora_A[lora_name])
            if (lora_name in layer.lora_B
                    and layer.lora_B[lora_name].requires_grad):
                params.append(layer.lora_B[lora_name])
        if (self.static_weight_logits is not None
                and self.static_weight_logits.requires_grad):
            params.append(self.static_weight_logits)
        return params

    def lora_weight_summary(self):
        if self.weight_mode != "residual_gate" or self.static_weight_logits is None:
            return None
        logits = self.static_weight_logits.detach().float().cpu()
        weights = 1.0 + 0.5 * torch.tanh(logits)
        return {
            dim: {
                "logit": float(logit),
                "weight": float(weight),
            }
            for dim, logit, weight in zip(
                self.gate_dimensions,
                logits.tolist(),
                weights.tolist(),
            )
        }

    def _weight_state(self):
        state = {
            "weight_mode": self.weight_mode,
            "weight_normalize": self.weight_normalize,
            "composition_mode": self.composition_mode,
            "dimension_order": self.dimension_order,
            "orthogonal_eps": self.orthogonal_eps,
            "orthogonal_strength": self.orthogonal_strength,
            "task_scale": self.task_scale,
            "knowledge_scale": self.knowledge_scale,
            "task_lora_name": self.task_lora_name,
            "lora_names": self.lora_names,
        }
        if self.gate_dimensions:
            state["gate_dimensions"] = self.gate_dimensions
            state["lora_name_to_gate_dimension"] = self.lora_name_to_gate_dimension
        if self.static_weight_logits is not None:
            state["static_weight_logits"] = (
                self.static_weight_logits.detach().cpu())
        summary = self.lora_weight_summary()
        if summary is not None:
            state["weight_summary"] = summary
        return state

    def save_weight_state(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        torch.save(self._weight_state(), os.path.join(base_dir, "lora_weighting.pt"))
        meta = {
            "weight_mode": self.weight_mode,
            "weight_normalize": self.weight_normalize,
            "composition_mode": self.composition_mode,
            "dimension_order": self.dimension_order,
            "orthogonal_eps": self.orthogonal_eps,
            "orthogonal_strength": self.orthogonal_strength,
            "task_scale": self.task_scale,
            "knowledge_scale": self.knowledge_scale,
            "task_lora_name": self.task_lora_name,
            "lora_names": self.lora_names,
        }
        if self.gate_dimensions:
            meta["gate_dimensions"] = self.gate_dimensions
            meta["lora_name_to_gate_dimension"] = self.lora_name_to_gate_dimension
        with open(os.path.join(base_dir, "lora_weighting_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_weight_state(self, base_dir: str):
        path = os.path.join(base_dir, "lora_weighting.pt")
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location="cpu")
        if state.get("weight_mode") != self.weight_mode:
            print(f"  [warn] saved weight_mode={state.get('weight_mode')} "
                  f"but current weight_mode={self.weight_mode}")
        if self.static_weight_logits is not None and "static_weight_logits" in state:
            saved = state["static_weight_logits"]
            if tuple(saved.shape) == tuple(self.static_weight_logits.shape):
                self.static_weight_logits.data.copy_(
                    saved.to(self.static_weight_logits.device))
            else:
                print(f"  [warn] saved weighting shape {tuple(saved.shape)} "
                      f"does not match current shape "
                      f"{tuple(self.static_weight_logits.shape)}; skipping")
        print(f"  Loaded LoRA weighting state from {public_path(path)}")

    # ----- save / load (PEFT compatible) -----
    def _uses_vllm_language_model_wrapper(self) -> bool:
        def _is_qwen_conditional_generation(cfg) -> bool:
            architectures = tuple(getattr(cfg, "architectures", None) or ())
            model_type = str(getattr(cfg, "model_type", "") or "")
            if any(arch in wrapper_architectures for arch in architectures):
                return True
            return (
                model_type in {"mistral3", "qwen3_5", "qwen3_5_moe"}
                and any("ForConditionalGeneration" in arch
                        for arch in architectures)
            )

        wrapper_architectures = {
            "Mistral3ForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }

        cfg = getattr(self.base_model, "config", None)
        if _is_qwen_conditional_generation(cfg):
            return True

        # Transformers AutoModelForCausalLM loads Qwen3.5 as the text-only
        # Qwen3_5ForCausalLM (module paths: model.layers.*), while vLLM
        # resolves the same local directory through its outer
        # Qwen3_5ForConditionalGeneration wrapper
        # (module paths: language_model.model.layers.*).  Inspect the
        # original model directory config so merged PEFT adapters get the
        # vLLM-facing prefix even when the in-memory HF model is text-only.
        cfg_path = os.path.join(str(self.model_name), "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    raw_cfg = json.load(f)
                raw_architectures = tuple(raw_cfg.get("architectures") or ())
                raw_model_type = str(raw_cfg.get("model_type") or "")
                if any(arch in wrapper_architectures
                       for arch in raw_architectures):
                    return True
                if (raw_model_type in {
                        "mistral3", "qwen3_5", "qwen3_5_moe"}
                        and any("ForConditionalGeneration" in arch
                                for arch in raw_architectures)):
                    return True
            except Exception as e:
                print(f"  [warn] could not inspect model config "
                      f"for vLLM prefix mapping: {e}")
        return False

    def _vllm_peft_prefix(self, path: str) -> str:
        """Return the PEFT key prefix matching vLLM's module names."""
        path = path.lstrip(".")
        cfg = getattr(self.base_model, "config", None)
        model_type = str(getattr(cfg, "model_type", "") or "")
        if model_type == "mistral3":
            if path.startswith("model.language_model."):
                suffix = path[len("model.language_model."):]
            elif path.startswith("language_model."):
                suffix = path[len("language_model."):]
            else:
                suffix = path
            if not suffix.startswith("model."):
                suffix = f"model.{suffix}"
            return f"base_model.model.language_model.{suffix}"
        if self._uses_vllm_language_model_wrapper():
            if path.startswith("model.language_model."):
                mapped_path = path[len("model."):]
            elif path.startswith("language_model."):
                mapped_path = path
            else:
                mapped_path = f"language_model.{path}"
        else:
            if path.startswith("model.language_model."):
                mapped_path = "model." + path[len("model.language_model."):]
            elif path.startswith("language_model."):
                mapped_path = path[len("language_model."):]
            else:
                mapped_path = path
        return f"base_model.model.{mapped_path}"

    def _peft_state_dict(self, lora_name: str) -> dict:
        tensors = {}
        for layer, path in zip(self.lora_layers, self.layer_paths):
            if lora_name not in layer.lora_A:
                continue
            prefix = f"base_model.model.{path}"
            tensors[f"{prefix}.lora_A.weight"] = (
                layer.lora_A[lora_name].data.cpu().clone())
            tensors[f"{prefix}.lora_B.weight"] = (
                layer.lora_B[lora_name].data.cpu().clone())
        return tensors

    def _peft_config(self, rank: int, alpha: int) -> dict:
        return {
            "auto_mapping": None,
            "base_model_name_or_path": public_path(self.model_name),
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "revision": None,
            "target_modules": self.target_modules,
            "task_type": "CAUSAL_LM",
        }

    def save_lora_peft(self, save_dir: str, lora_name: str):
        os.makedirs(save_dir, exist_ok=True)
        tensors = self._peft_state_dict(lora_name)
        if HAS_SAFETENSORS:
            safetensors_save(tensors,
                             os.path.join(save_dir, "adapter_model.safetensors"))
        else:
            torch.save(tensors,
                       os.path.join(save_dir, "adapter_model.bin"))
        cfg = self._peft_config(self.rank, self.alpha)
        with open(os.path.join(save_dir, "adapter_config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)

    def save_all_loras(self, base_dir: str):
        for name in self.lora_names:
            self.save_lora_peft(os.path.join(base_dir, name), name)
        self.save_weight_state(base_dir)
        print(f"Saved {len(self.lora_names)} adapters -> "
              f"{public_path(base_dir)}")

    def load_lora(self, load_dir: str, lora_name: str):
        st = os.path.join(load_dir, "adapter_model.safetensors")
        bn = os.path.join(load_dir, "adapter_model.bin")
        if os.path.exists(st) and HAS_SAFETENSORS:
            state = safetensors_load(st)
        elif os.path.exists(bn):
            state = torch.load(bn, map_location='cpu')
        else:
            print(f"  [warn] adapter not found at {public_path(load_dir)}")
            return
        loaded = 0
        for layer, path in zip(self.lora_layers, self.layer_paths):
            ka = f"base_model.model.{path}.lora_A.weight"
            kb = f"base_model.model.{path}.lora_B.weight"
            ka_legacy = f"base_model.model.{path}.lora_A.default.weight"
            kb_legacy = f"base_model.model.{path}.lora_B.default.weight"
            if ka_legacy in state and ka not in state:
                ka, kb = ka_legacy, kb_legacy
            if ka in state and lora_name in layer.lora_A:
                dev = layer.lora_A[lora_name].device
                layer.lora_A[lora_name].data.copy_(state[ka].to(dev))
                layer.lora_B[lora_name].data.copy_(state[kb].to(dev))
                loaded += 1
        print(f"  Loaded '{lora_name}' ({loaded} layers)")

    def load_all_loras(self, base_dir: str):
        for name in self.lora_names:
            d = os.path.join(base_dir, name)
            if os.path.exists(d):
                self.load_lora(d, name)
            else:
                print(f"  [skip] '{name}' not found at {d}")
        self.load_weight_state(base_dir)

    # ----- merge multiple LoRAs into one PEFT adapter -----
    def merge_and_save(self, active_names: List[str], save_dir: str,
                       weights: Optional[Dict[str, float]] = None):
        """Concatenate several LoRA adapters into one PEFT adapter.

        The trick: cat A matrices along dim-0 and B along dim-1 so that
        B_merged @ A_merged  =  sum_i  B_i @ A_i
        Merged rank = N * rank, merged alpha = N * alpha  (preserves scaling).

        Keys use a vLLM-compatible prefix. Some vLLM model classes expose
        transformer blocks under language_model, while LLaMA exposes them
        directly under model.
        """
        os.makedirs(save_dir, exist_ok=True)
        names = self._order_lora_names(active_names)
        if weights is None and self.weight_mode in ("static", "residual_gate"):
            learned = self.compute_lora_weights(names)
            if learned is not None:
                weight_names = names
                if self.composition_mode == "explicit_task_knowledge_projection":
                    weight_names = [
                        name for name in names
                        if name != self.task_lora_name
                    ]
                weights = {
                    name: float(learned[i].detach().cpu())
                    for i, name in enumerate(weight_names)
                }
        n = len(names)
        merged_rank = n * self.rank
        merged_alpha = n * self.alpha

        tensors = {}
        for layer, path in zip(self.lora_layers, self.layer_paths):
            As = [layer.lora_A[nm].data for nm in names
                  if nm in layer.lora_A]
            coeffs = layer.merge_coefficients(names, weights=weights)
            Bs = []
            for nm in names:
                if nm not in layer.lora_B:
                    continue
                w = float(coeffs.get(nm, 1.0))
                Bs.append(layer.lora_B[nm].data * w)
            if not As:
                continue
            A_cat = torch.cat(As, dim=0).cpu()   # [N*r, in]
            B_cat = torch.cat(Bs, dim=1).cpu()   # [out, N*r]
            prefix = self._vllm_peft_prefix(path)
            tensors[f"{prefix}.lora_A.weight"] = A_cat
            tensors[f"{prefix}.lora_B.weight"] = B_cat

        if HAS_SAFETENSORS:
            safetensors_save(
                tensors,
                os.path.join(save_dir, "adapter_model.safetensors"))
        else:
            torch.save(tensors,
                       os.path.join(save_dir, "adapter_model.bin"))

        cfg = self._peft_config(merged_rank, merged_alpha)
        with open(os.path.join(save_dir, "adapter_config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"  Merged {n} LoRAs (r={merged_rank}, a={merged_alpha}) "
              f"-> {public_path(save_dir)}")
