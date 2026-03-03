import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_resume import atomic_torch_save
from models.super_nets.super_proxyless import SuperProxylessNASNets


ArchParamType = List[torch.Tensor]


@dataclass
class ClientHistory:
    task_prototypes: Dict[int, torch.Tensor] = field(default_factory=dict)
    arch_parameters: Dict[int, ArchParamType] = field(default_factory=dict)
    subnet_artifacts: Dict[int, str] = field(default_factory=dict)

    def has_any(self) -> bool:
        has_proto = any(proto is not None for proto in self.task_prototypes.values())
        has_arch = any(params is not None for params in self.arch_parameters.values())
        return has_proto and has_arch

    def export_state(self) -> dict:
        return {
            "task_prototypes": {
                int(task_id): proto.detach().cpu().clone()
                for task_id, proto in self.task_prototypes.items()
                if proto is not None
            },
            "arch_parameters": {
                int(task_id): [tensor.detach().cpu().clone() for tensor in tensors]
                for task_id, tensors in self.arch_parameters.items()
                if tensors is not None
            },
            "subnet_artifacts": {
                int(task_id): path
                for task_id, path in self.subnet_artifacts.items()
                if path is not None
            },
        }

    @classmethod
    def from_state(cls, state: Optional[dict]):
        history = cls()
        if not state:
            return history
        history.task_prototypes = {
            int(task_id): proto.detach().cpu().clone()
            for task_id, proto in state.get("task_prototypes", {}).items()
            if proto is not None
        }
        history.arch_parameters = {
            int(task_id): [tensor.detach().cpu().clone() for tensor in tensors]
            for task_id, tensors in state.get("arch_parameters", {}).items()
            if tensors is not None
        }
        history.subnet_artifacts = {
            int(task_id): path
            for task_id, path in state.get("subnet_artifacts", {}).items()
            if path is not None
        }
        return history


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def clone_arch_parameters(model) -> ArchParamType:
    base_model = unwrap_model(model)
    if not hasattr(base_model, "architecture_parameters"):
        return []
    return [param.detach().cpu().clone() for param in base_model.architecture_parameters()]


def summarize_arch_parameters(arch_params: ArchParamType) -> dict:
    if not arch_params:
        return {
            "alpha_param_count": 0,
            "alpha_before_mean": 0.0,
            "alpha_before_std": 0.0,
        }
    flat = torch.cat([tensor.detach().float().reshape(-1).cpu() for tensor in arch_params])
    return {
        "alpha_param_count": int(flat.numel()),
        "alpha_mean": float(flat.mean().item()),
        "alpha_std": float(flat.std(unbiased=False).item()),
        "alpha_norm": float(flat.norm().item()),
    }


def arch_parameter_delta_norm(before_params: ArchParamType, after_params: ArchParamType) -> float:
    if not before_params or not after_params:
        return 0.0
    before_flat = torch.cat([tensor.detach().float().reshape(-1).cpu() for tensor in before_params])
    after_flat = torch.cat([tensor.detach().float().reshape(-1).cpu() for tensor in after_params])
    return float((after_flat - before_flat).norm().item())


def compute_task_prototype(model, dataloader, device) -> Optional[torch.Tensor]:
    base_model = unwrap_model(model)
    if not all(hasattr(base_model, attr) for attr in ["first_conv", "blocks", "feature_mix_layer", "global_avg_pooling"]):
        raise ValueError(f"Model {type(base_model).__name__} does not expose Proxyless-style GAP features")

    was_training = base_model.training
    base_model = base_model.to(device)
    base_model.eval()
    feature_sum = None
    sample_count = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            x = base_model.first_conv(images)
            for block in base_model.blocks:
                x = block(x)
            x = base_model.feature_mix_layer(x)
            x = base_model.global_avg_pooling(x)
            x = x.view(x.size(0), -1)
            batch_sum = x.detach().float().sum(dim=0).cpu()
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            sample_count += int(x.size(0))

    if was_training:
        base_model.train()

    if sample_count == 0 or feature_sum is None:
        return None
    return feature_sum / float(sample_count)


def compute_similarity_scores(current_proto: torch.Tensor, history_protos: Dict[int, torch.Tensor]) -> Dict[int, float]:
    if current_proto is None:
        return {}
    current_vec = F.normalize(current_proto.detach().float().reshape(1, -1), p=2, dim=1)
    similarity_scores = {}
    for task_id, proto in history_protos.items():
        if proto is None:
            continue
        proto_vec = F.normalize(proto.detach().float().reshape(1, -1), p=2, dim=1)
        similarity_scores[int(task_id)] = float((current_vec * proto_vec).sum(dim=1).item())
    return similarity_scores


def build_arch_prior_weights(
    current_proto: torch.Tensor,
    history_protos: Dict[int, torch.Tensor],
    topk: int,
    tau: float,
) -> Tuple[List[int], Optional[torch.Tensor]]:
    similarity_scores = compute_similarity_scores(current_proto, history_protos)
    if len(similarity_scores) == 0:
        return [], None

    sorted_pairs = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    selected_pairs = sorted_pairs[: max(1, min(int(topk), len(sorted_pairs)))]
    selected_task_ids = [task_id for task_id, _ in selected_pairs]
    score_tensor = torch.tensor([score for _, score in selected_pairs], dtype=torch.float32)
    tau_value = float(tau) if float(tau) > 0 else 1.0
    weights = torch.softmax(score_tensor / tau_value, dim=0)
    return selected_task_ids, weights


def build_fused_arch_parameters(
    history_arch_params: Dict[int, ArchParamType],
    selected_task_ids: List[int],
    weights: torch.Tensor,
) -> Optional[ArchParamType]:
    if not selected_task_ids or weights is None:
        return None
    reference = history_arch_params.get(int(selected_task_ids[0]))
    if reference is None:
        return None
    fused_params = []
    for param_index in range(len(reference)):
        fused_tensor = None
        for weight, task_id in zip(weights, selected_task_ids):
            hist_params = history_arch_params.get(int(task_id))
            if hist_params is None or len(hist_params) <= param_index:
                continue
            contrib = hist_params[param_index].detach().float().cpu() * float(weight.item())
            fused_tensor = contrib if fused_tensor is None else fused_tensor + contrib
        if fused_tensor is None:
            return None
        fused_params.append(fused_tensor)
    return fused_params


def init_arch_parameters_with_prior(current_model, history_arch_params, selected_task_ids, weights):
    fused_params = build_fused_arch_parameters(history_arch_params, selected_task_ids, weights)
    if fused_params is None:
        return None

    base_model = unwrap_model(current_model)
    current_params = list(base_model.architecture_parameters()) if hasattr(base_model, "architecture_parameters") else []
    if not current_params:
        return None

    before_params = clone_arch_parameters(base_model)
    with torch.no_grad():
        for current_param, fused_param in zip(current_params, fused_params):
            current_param.copy_(fused_param.to(device=current_param.device, dtype=current_param.dtype))
    after_params = clone_arch_parameters(base_model)

    before_stats = summarize_arch_parameters(before_params)
    after_stats = summarize_arch_parameters(after_params)
    return {
        "fused_arch_params": [tensor.detach().cpu().clone() for tensor in fused_params],
        "alpha_before_mean": before_stats.get("alpha_mean", 0.0),
        "alpha_before_std": before_stats.get("alpha_std", 0.0),
        "alpha_after_mean": after_stats.get("alpha_mean", 0.0),
        "alpha_after_std": after_stats.get("alpha_std", 0.0),
        "alpha_diff_norm": arch_parameter_delta_norm(before_params, after_params),
        "alpha_before_norm": before_stats.get("alpha_norm", 0.0),
        "alpha_after_norm": after_stats.get("alpha_norm", 0.0),
    }


def apply_fused_arch_parameters(current_model, fused_arch_params: ArchParamType):
    if not fused_arch_params:
        return None
    base_model = unwrap_model(current_model)
    current_params = list(base_model.architecture_parameters()) if hasattr(base_model, "architecture_parameters") else []
    if not current_params:
        return None
    before_params = clone_arch_parameters(base_model)
    with torch.no_grad():
        for current_param, fused_param in zip(current_params, fused_arch_params):
            current_param.copy_(fused_param.to(device=current_param.device, dtype=current_param.dtype))
    after_params = clone_arch_parameters(base_model)
    before_stats = summarize_arch_parameters(before_params)
    after_stats = summarize_arch_parameters(after_params)
    return {
        "alpha_before_mean": before_stats.get("alpha_mean", 0.0),
        "alpha_before_std": before_stats.get("alpha_std", 0.0),
        "alpha_after_mean": after_stats.get("alpha_mean", 0.0),
        "alpha_after_std": after_stats.get("alpha_std", 0.0),
        "alpha_diff_norm": arch_parameter_delta_norm(before_params, after_params),
        "alpha_before_norm": before_stats.get("alpha_norm", 0.0),
        "alpha_after_norm": after_stats.get("alpha_norm", 0.0),
    }


def save_client_histories(histories: List[ClientHistory], save_path: str):
    states = [history.export_state() for history in histories]
    atomic_torch_save(states, save_path)


def load_client_histories(load_path: str, num_clients: int) -> List[ClientHistory]:
    histories = [ClientHistory() for _ in range(num_clients)]
    if not os.path.isfile(load_path):
        return histories
    state_list = torch.load(load_path, map_location="cpu")
    if not isinstance(state_list, list):
        return histories
    for idx, state in enumerate(state_list[:num_clients]):
        histories[idx] = ClientHistory.from_state(state)
    return histories


def export_supernet_client_subnet(search_manager, artifact_path: str, supernet_kwargs: dict):
    export_kwargs = dict(supernet_kwargs)
    export_kwargs["width_stages"] = list(export_kwargs.get("width_stages", []))
    export_kwargs["n_cell_stages"] = list(export_kwargs.get("n_cell_stages", []))
    export_kwargs["stride_stages"] = list(export_kwargs.get("stride_stages", []))
    export_kwargs["conv_candidates"] = list(export_kwargs.get("conv_candidates", []))

    supernet_copy = SuperProxylessNASNets(**export_kwargs).cpu()
    source_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in search_manager.net.state_dict().items()
    }
    supernet_copy.load_state_dict(source_state_dict, strict=True)
    normal_net = supernet_copy.convert_to_normal_net()
    payload = {
        "net_config": normal_net.config,
        "state_dict": normal_net.state_dict(),
    }
    atomic_torch_save(payload, artifact_path)
    return normal_net


def load_subnet_from_artifact(artifact_path: str):
    if not artifact_path or not os.path.isfile(artifact_path):
        return None
    from models import get_net_by_name

    payload = torch.load(artifact_path, map_location="cpu")
    net_config = payload.get("net_config")
    if net_config is None:
        return None
    subnet = get_net_by_name(net_config["name"]).build_from_config(net_config)
    state_dict = payload.get("state_dict", {})
    subnet.load_state_dict(state_dict, strict=False)
    return subnet


def append_arch_prior_log(log_path: str, record: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=True) + "\n")
