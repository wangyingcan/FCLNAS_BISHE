"""
Legacy forgetting helpers for NAS search.

These utilities preserve the previous "search stage + forgetting mitigation"
implementation for ablation or regression studies. The active ProxylessNAS
search pipeline no longer calls them: search loss now only uses the current
task objective plus NAS complexity regularization.
"""

import copy

import torch
import torch.nn.functional as F

from utils_old import cross_entropy_with_label_smoothing


def prepare_distill_model(search_manager, teacher_model):
    if teacher_model is None:
        return None
    model = copy.deepcopy(teacher_model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(search_manager.run_manager.device)
    model.eval()
    return model


def prepare_reg_anchor(search_manager, teacher_model, reg_lambda, reg_use_ewc):
    anchor_params = None
    fisher = None
    if teacher_model is None or reg_lambda <= 0:
        return anchor_params, fisher

    reg_model = copy.deepcopy(teacher_model)
    if isinstance(reg_model, torch.nn.DataParallel):
        reg_model = reg_model.module
    reg_model = reg_model.to(search_manager.run_manager.device)
    reg_model.eval()

    anchor_params = {}
    for name, param in reg_model.named_parameters():
        if not param.requires_grad:
            continue
        anchor_params[name.replace("module.", "", 1)] = param.detach().clone()

    if reg_use_ewc:
        fisher = {}
        sample_loader = list(search_manager.run_manager.run_config.train_loader)
        if sample_loader:
            images, labels = sample_loader[0]
            images = images.to(search_manager.run_manager.device)
            labels = labels.to(search_manager.run_manager.device)
            reg_model.zero_grad()
            out = reg_model(images)
            ce = F.cross_entropy(out, labels)
            ce.backward()
            for name, param in reg_model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher[name.replace("module.", "", 1)] = param.grad.detach().pow(2)
    return anchor_params, fisher


def build_total_loss_for_search(
    search_manager,
    output,
    labels,
    teacher_output,
    eff_kd_method,
    eff_kd_lambda,
    eff_kd_temperature,
    eff_kd_conf,
    reg_anchor_params,
    reg_lambda,
    fisher_params,
    cl_penalty_clip,
):
    if search_manager.run_manager.run_config.label_smoothing > 0:
        ce_loss = cross_entropy_with_label_smoothing(
            output, labels, search_manager.run_manager.run_config.label_smoothing
        )
    else:
        ce_loss = search_manager.run_manager.criterion(output, labels)

    ewc_penalty = search_manager.run_manager._ewc_penalty()
    kd_loss = None
    if teacher_output is not None and eff_kd_method in ["logit", "logit_conf"] and eff_kd_lambda > 0:
        temp = eff_kd_temperature if eff_kd_temperature is not None else 1.0
        student_logp = F.log_softmax(output / temp, dim=1)
        teacher_prob = F.softmax(teacher_output / temp, dim=1)
        if eff_kd_method == "logit":
            kd_loss = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (temp * temp)
        else:
            mask = (teacher_prob > eff_kd_conf).float()
            if mask.sum() > 0:
                log_teacher = torch.log(teacher_prob + 1e-12)
                kd_loss = ((teacher_prob * (log_teacher - student_logp)) * mask).sum() / mask.sum()
                kd_loss = kd_loss * (temp * temp)

    reg_loss = None
    if reg_anchor_params is not None and reg_lambda > 0:
        reg_loss = 0.0
        for name, param in search_manager.run_manager.net.named_parameters():
            if not param.requires_grad:
                continue
            norm_name = name.replace("module.", "", 1)
            anchor = reg_anchor_params.get(norm_name)
            if anchor is None or anchor.shape != param.shape:
                continue
            weight = fisher_params.get(norm_name, 1.0) if fisher_params is not None else 1.0
            if torch.is_tensor(weight):
                weight = weight.mean().item()
            reg_loss = reg_loss + weight * (param - anchor).pow(2).sum()

    total_loss = ce_loss
    if kd_loss is not None and eff_kd_lambda > 0:
        total_loss = total_loss + eff_kd_lambda * kd_loss
    if ewc_penalty is not None and torch.isfinite(ewc_penalty):
        penalty_term = search_manager.run_manager.ewc_lambda * ewc_penalty
        if cl_penalty_clip is not None:
            penalty_term = torch.clamp(penalty_term, max=cl_penalty_clip)
        total_loss = total_loss + penalty_term
    if reg_loss is not None and reg_lambda > 0:
        penalty_term = reg_lambda * reg_loss
        if cl_penalty_clip is not None:
            penalty_term = torch.clamp(penalty_term, max=cl_penalty_clip)
        total_loss = total_loss + penalty_term
    return total_loss, kd_loss, reg_loss, ewc_penalty
