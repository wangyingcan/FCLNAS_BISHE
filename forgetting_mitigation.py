import torch
import torch.nn.functional as F


def replay_is_enabled(manager) -> bool:
    return (
        bool(getattr(manager, "enable_replay", False))
        and getattr(manager, "replay_mode", "none") != "none"
        and getattr(manager, "replay_capacity", 0) > 0
        and getattr(manager, "replay_buffer", None) is not None
    )


def ewc_is_enabled(manager) -> bool:
    return bool(getattr(manager, "enable_ewc", False)) and getattr(manager, "ewc_lambda", 0.0) > 0


def orthogonal_update_is_enabled(manager) -> bool:
    return (
        bool(getattr(manager, "enable_orthogonal_update", False))
        and getattr(manager, "cl_ortho_method", "none") != "none"
    )


def apply_replay_loss(manager, images, labels, round_data_meter=None):
    if not replay_is_enabled(manager):
        return images, labels

    mix_allowed = (
        manager.replay_per_batch > 0
        and len(manager.replay_buffer) > 0
        and (
            not getattr(manager, "replay_only_training", False)
            or getattr(manager, "allow_mix_during_stage", False)
        )
    )
    if not mix_allowed:
        return images, labels

    rep_x, rep_y = manager.replay_buffer.sample(
        manager.replay_per_batch,
        mode=manager.replay_mode,
        exclude_task=None if manager.replay_mode == "global" else None,
        current_task=manager.task_id,
        old_task_scale=manager.replay_old_task_scale,
        forgetting_map=getattr(manager, "task_forgetting", None),
        old_task_scale_by_f=manager.replay_old_task_scale_by_f,
    )
    if rep_x is None or rep_y is None or rep_x.numel() == 0:
        return images, labels

    if round_data_meter is not None:
        rep_count = rep_x.size(0)
        round_data_meter["samples_from_replay"] = round_data_meter.get(
            "samples_from_replay", 0
        ) + rep_count
        round_data_meter["mixed_batches"] = round_data_meter.get("mixed_batches", 0) + 1
        manager._accumulate_label_hist(rep_y, round_data_meter.get("replay_task_hist"))

    rep_x = rep_x.to(manager.device, non_blocking=True)
    rep_y = rep_y.to(manager.device, non_blocking=True)
    return torch.cat([images, rep_x], dim=0), torch.cat([labels, rep_y], dim=0)


def apply_ewc_regularization(manager):
    if not ewc_is_enabled(manager):
        return None
    penalty = manager._ewc_penalty()
    if penalty is None or not torch.isfinite(penalty):
        return None
    return penalty


def apply_orthogonal_update(manager, total_loss, kd_loss=None):
    if not orthogonal_update_is_enabled(manager):
        manager.optimizer.zero_grad()
        manager.net.zero_grad()
        total_loss.backward()
        return None, None

    cos_before, cos_after = None, None
    if (
        manager.cl_ortho_method == "kd_ortho"
        and manager.teacher_model is not None
        and manager.cl_kd_logit_lambda > 0
        and kd_loss is not None
    ):
        old_grads = None
        if manager.kd_ortho_ref_grads is not None:
            old_grads = {
                name: grad.to(manager.device)
                for name, grad in manager.kd_ortho_ref_grads.items()
            }
        else:
            manager.optimizer.zero_grad()
            manager.net.zero_grad()
            kd_loss.backward(retain_graph=True)
            old_grads = {}
            for name, param in manager.net.module.named_parameters():
                if param.grad is not None and param.requires_grad:
                    old_grads[name] = param.grad.detach().clone()
            manager.optimizer.zero_grad()
            manager.net.zero_grad()

        manager.optimizer.zero_grad()
        manager.net.zero_grad()
        total_loss.backward()

        eps = 1e-12
        g_list_before = []
        g_old_list = []
        for name, param in manager.net.module.named_parameters():
            if param.grad is None or name not in old_grads:
                continue
            grad = param.grad
            grad_old = old_grads[name].to(grad.device)

            grad_flat = grad.view(-1)
            grad_old_flat = grad_old.view(-1)
            denom = torch.dot(grad_old_flat, grad_old_flat) + eps
            if denom.item() == 0.0:
                continue

            dot = torch.dot(grad_flat, grad_old_flat)
            proj_coef = dot / denom

            g_list_before.append(grad_flat.detach().clone())
            g_old_list.append(grad_old_flat.detach().clone())

            scale = manager.cl_ortho_scale
            grad_ortho_flat = grad_flat - scale * proj_coef * grad_old_flat
            param.grad.copy_(grad_ortho_flat.view_as(param))

        if g_list_before and g_old_list:
            grad_flat_all = torch.cat(g_list_before)
            grad_old_all = torch.cat(g_old_list)
            cos_before = F.cosine_similarity(
                grad_flat_all.unsqueeze(0), grad_old_all.unsqueeze(0), dim=1
            ).item()
            g_list_after = []
            for name, param in manager.net.module.named_parameters():
                if param.grad is None or name not in old_grads:
                    continue
                g_list_after.append(param.grad.detach().view(-1))
            if g_list_after:
                grad_flat_after = torch.cat(g_list_after)
                cos_after = F.cosine_similarity(
                    grad_flat_after.unsqueeze(0), grad_old_all.unsqueeze(0), dim=1
                ).item()
    elif manager.cl_ortho_method == "prev_grad_ortho":
        manager.optimizer.zero_grad()
        manager.net.zero_grad()
        total_loss.backward()
        cos_before, cos_after = manager.apply_ortho_projection_with_previous_gradients(
            return_cos=True, use_kd=False
        )
        manager.save_gradients()
    elif manager.cl_ortho_method == "kd_prev_grad_ortho" and manager.teacher_model is not None:
        manager.optimizer.zero_grad()
        manager.net.zero_grad()
        total_loss.backward()
        if manager.prev_kd_grads is not None:
            cos_before, cos_after = manager.apply_ortho_projection_with_previous_gradients(
                return_cos=True, use_kd=True, use_history=False
            )
    else:
        manager.optimizer.zero_grad()
        manager.net.zero_grad()
        total_loss.backward()
        cos_before, cos_after = manager.apply_ortho_projection(return_cos=True)

    return cos_before, cos_after
