import torch


def replay_is_enabled(manager) -> bool:
    return (
        bool(getattr(manager, "enable_replay", False))
        and getattr(manager, "replay_mode", "none") != "none"
        and getattr(manager, "replay_capacity", 0) > 0
        and getattr(manager, "replay_buffer", None) is not None
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
