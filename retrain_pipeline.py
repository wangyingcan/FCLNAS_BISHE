import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List

import torch

from auto_resume import atomic_torch_save
from commonwealth_machine import CommonwealthMachine
from models import get_net_by_name
from models.baseline_nets import BaselineResNet
from run_manager import CifarRunConfig, RunManager, SimpleReplayBuffer


@dataclass
class RetrainPipelineHelpers:
    attach_replay_cfg: Callable
    load_prev_task: Callable
    save_replay_buffers: Callable
    load_model_from_checkpoint: Callable
    restore_retrain_bootstrap: Callable
    load_teacher_snapshot_like: Callable
    model_signature: Callable
    train_personalized_subnet: Callable

def teacher_is_required(args) -> bool:
    kd_enabled = bool(getattr(args, "enable_kd", False))
    kd_weight = float(getattr(args, "cl_kd_logit_lambda", 0.0))
    kd_method = str(getattr(args, "cl_kd_method", "none")).lower()
    use_kd_loss = kd_enabled and kd_weight > 0 and kd_method in {"logit", "logit_conf"}
    return use_kd_loss


def _retrain_total_epochs(args, retrain_last_round: int) -> int:
    return max(1, int(getattr(args, "local_epoch_number", 1)) * int(retrain_last_round))


def _build_retrain_run_config_kwargs(args, retrain_last_round: int):
    cfg = dict(args.__dict__)
    cfg["n_epochs"] = _retrain_total_epochs(args, retrain_last_round)
    cfg["search"] = False
    return cfg


def _prepare_retrain_run_config(
    run_config: CifarRunConfig,
    retrain_last_round: int,
    local_epoch_number: int = 1,
):
    run_config.search = False
    run_config.n_epochs = max(1, int(local_epoch_number) * int(retrain_last_round))
    return run_config


def _reset_run_manager_task(run_mgr: RunManager, new_task_id: int):
    run_mgr.task_id = new_task_id
    run_mgr.run_config.task_id = new_task_id
    run_mgr.run_config.search = False
    run_mgr.run_config._data_provider = None
    run_mgr.run_config._train_iter = None
    run_mgr.run_config._valid_iter = None
    run_mgr.run_config._test_iter = None


def _rebuild_replay_buffer(run_mgr: RunManager, tasks_to_fill: List[int]):
    buf = getattr(run_mgr, "replay_buffer", None)
    if (
        buf is None
        or getattr(run_mgr, "replay_mode", "none") == "none"
        or getattr(run_mgr, "replay_capacity", 0) <= 0
        or not tasks_to_fill
    ):
        return {}
    buf.clear()
    stats = {}
    prev_task = run_mgr.task_id
    quota = buf.capacity // len(tasks_to_fill) if buf.capacity > 0 else 0
    for task_to_fill in tasks_to_fill:
        _reset_run_manager_task(run_mgr, task_to_fill)
        added = 0
        for images, labels in run_mgr.run_config.train_loader:
            buf.add_batch(images.detach().cpu(), labels.detach().cpu(), task_to_fill)
            added += images.size(0)
            if quota > 0 and added >= quota:
                break
            if buf.capacity > 0 and len(buf) >= buf.capacity:
                break
        stats[task_to_fill] = added
        if buf.capacity > 0 and len(buf) >= buf.capacity:
            break
    _reset_run_manager_task(run_mgr, prev_task)
    hist = buf.task_hist() if hasattr(buf, "task_hist") else {}
    print(f"[ReplayPrefill] tasks={tasks_to_fill} quota={quota} buffer_size={len(buf)} hist={hist}")
    return stats


def _build_stage_training_buffer(run_mgr: RunManager, task_to_fill: int):
    main_buf = getattr(run_mgr, "replay_buffer", None)
    storage = getattr(main_buf, "storage", []) if main_buf is not None else []
    task_entries = [entry for entry in storage if int(entry.get("task", -1)) == int(task_to_fill)]
    if not task_entries:
        print(f"[StageBuffer] task={task_to_fill} 无可用样本，stage_buffer 将为空")
        return None
    stage_buf = SimpleReplayBuffer(len(task_entries))
    stage_buf.storage = [
        {
            "x": entry["x"].clone(),
            "y": entry["y"].clone(),
            "task": int(entry.get("task", task_to_fill)),
            "t": int(entry.get("t", 0)),
        }
        for entry in task_entries
    ]
    stage_buf._time = max(entry.get("t", 0) for entry in task_entries)
    hist = stage_buf.task_hist() if hasattr(stage_buf, "task_hist") else {}
    print(f"[StageBuffer] task={task_to_fill} stage_size={len(stage_buf)} hist={hist}")
    return stage_buf


def _load_client_subnet_artifact(artifact_path: str):
    if not artifact_path or not os.path.isfile(artifact_path):
        return None, None
    payload = torch.load(artifact_path, map_location="cpu")
    net_config = payload.get("net_config")
    if net_config is None:
        return None, None
    subnet = get_net_by_name(net_config["name"]).build_from_config(net_config)
    state_dict = payload.get("state_dict", {})
    subnet.load_state_dict(state_dict, strict=False)
    return subnet, net_config


def _normalize_baseline_method(method_name: str) -> str:
    method = str(method_name or "fedavg").strip().lower().replace("-", "_")
    aliases = {
        "fedavg": "fedavg",
        "target": "target",
        "refed": "re_fed",
        "re_fed": "re_fed",
        "ditto": "ditto",
        "fedweit": "fedweit",
        "fedta": "fedta",
        "affcl": "af_fcl",
        "af_fcl": "af_fcl",
    }
    if method not in aliases:
        raise ValueError(
            f"unsupported baseline_method={method_name}, expected one of: "
            f"{', '.join(sorted(set(aliases.values())))}"
        )
    return aliases[method]


def _configure_baseline_method(args):
    method = _normalize_baseline_method(getattr(args, "baseline_method", "fedavg"))
    auto_cfg = bool(getattr(args, "baseline_auto_config", True))
    args.baseline_method = method
    # baseline 对照组均采用 FedAvg 全局聚合（个性化方法在其上做附加约束）
    args.retrain_fedavg = True

    if method in {"re_fed", "target"} and auto_cfg:
        args.enable_replay = True
        if str(getattr(args, "replay_mode", "none")).lower() == "none":
            args.replay_mode = str(getattr(args, "baseline_replay_mode", "task_balanced"))
        if (
            int(getattr(args, "replay_capacity", 0)) <= 0
            and getattr(args, "replay_capacity_ratio", None) is None
        ):
            args.replay_capacity_ratio = float(getattr(args, "baseline_replay_capacity_ratio", 0.1))
        if int(getattr(args, "replay_per_batch", 0)) <= 0:
            args.replay_per_batch = int(getattr(args, "baseline_replay_per_batch", 32))

    if method == "target" and auto_cfg:
        args.enable_kd = True
        if str(getattr(args, "cl_kd_method", "none")).lower() == "none":
            args.cl_kd_method = str(getattr(args, "baseline_target_kd_method", "logit"))
        if float(getattr(args, "cl_kd_logit_lambda", 0.0)) <= 0:
            args.cl_kd_logit_lambda = float(getattr(args, "baseline_target_kd_lambda", 1.0))
        if float(getattr(args, "cl_kd_temperature", 0.0)) <= 0:
            args.cl_kd_temperature = float(getattr(args, "baseline_target_kd_temperature", 2.0))

    if method == "ditto":
        args.ditto_mu = float(getattr(args, "ditto_mu", 0.01))
        if args.ditto_mu <= 0:
            raise ValueError("ditto_mu must be > 0 when baseline_method=ditto")

    if method == "fedta":
        args.fedta_tail_ratio = float(getattr(args, "fedta_tail_ratio", 0.4))
        args.fedta_anchor_lambda = float(getattr(args, "fedta_anchor_lambda", 0.5))
        args.fedta_temperature = float(getattr(args, "fedta_temperature", 2.0))
        args.fedta_min_tail_classes = int(getattr(args, "fedta_min_tail_classes", 1))
        if args.fedta_anchor_lambda <= 0:
            raise ValueError("fedta_anchor_lambda must be > 0 when baseline_method=fedta")
        if args.fedta_temperature <= 0:
            raise ValueError("fedta_temperature must be > 0 when baseline_method=fedta")
        if args.fedta_min_tail_classes <= 0:
            raise ValueError("fedta_min_tail_classes must be >= 1 when baseline_method=fedta")

    if method == "af_fcl" and auto_cfg:
        # AF-FCL（工程复现版）：遗忘感知回放 + 目标蒸馏
        args.enable_replay = True
        if str(getattr(args, "replay_mode", "none")).lower() == "none":
            args.replay_mode = str(getattr(args, "baseline_affcl_replay_mode", "age_priority"))
        if (
            int(getattr(args, "replay_capacity", 0)) <= 0
            and getattr(args, "replay_capacity_ratio", None) is None
        ):
            args.replay_capacity_ratio = float(getattr(args, "baseline_affcl_replay_capacity_ratio", 0.15))
        if int(getattr(args, "replay_per_batch", 0)) <= 0:
            args.replay_per_batch = int(getattr(args, "baseline_affcl_replay_per_batch", 32))
        if float(getattr(args, "replay_old_task_scale", 1.0)) <= 1.0:
            args.replay_old_task_scale = float(getattr(args, "baseline_affcl_old_task_scale", 1.2))
        if float(getattr(args, "replay_old_task_scale_by_F", 0.0)) <= 0.0:
            args.replay_old_task_scale_by_F = float(getattr(args, "baseline_affcl_old_task_scale_by_f", 2.0))

        args.enable_kd = True
        if str(getattr(args, "cl_kd_method", "none")).lower() == "none":
            args.cl_kd_method = str(getattr(args, "baseline_affcl_kd_method", "logit"))
        if float(getattr(args, "cl_kd_logit_lambda", 0.0)) <= 0:
            args.cl_kd_logit_lambda = float(getattr(args, "baseline_affcl_kd_lambda", 0.5))
        if float(getattr(args, "cl_kd_temperature", 0.0)) <= 0:
            args.cl_kd_temperature = float(getattr(args, "baseline_affcl_kd_temperature", 2.0))

    return method


def _configure_supernet_retrain_replay(args, task_id: int):
    """
    NAS 子网重训默认启用历史样本回放（task>1），避免忘记显式传参导致无回放。
    若用户已手动指定 replay 参数，则尊重用户配置。
    """
    if int(task_id) <= 1:
        return
    if bool(getattr(args, "enable_replay", False)):
        return

    args.enable_replay = True
    if str(getattr(args, "replay_mode", "none")).lower() == "none":
        args.replay_mode = "task_balanced"
    if int(getattr(args, "replay_per_batch", 0)) <= 0:
        train_bs = int(getattr(args, "train_batch_size", 128))
        args.replay_per_batch = max(16, min(64, train_bs // 4))
    if int(getattr(args, "replay_capacity", 0)) <= 0 and getattr(args, "replay_capacity_ratio", None) is None:
        args.replay_capacity_ratio = 0.1
    print(
        "[RetrainReplay] auto-enable replay for NAS retrain: "
        f"mode={args.replay_mode}, per_batch={args.replay_per_batch}, "
        f"capacity={getattr(args, 'replay_capacity', 0)}, "
        f"capacity_ratio={getattr(args, 'replay_capacity_ratio', None)}"
    )


def run_supernet_retrain_pipeline(
    args,
    task_id: int,
    prev_task_path: str,
    current_task_path: str,
    global_server,
    clients_run_config_arr,
    replay_buffers_across_tasks,
    runtime_context,
    auto_resume_manager,
    user_resume: bool,
    helpers: RetrainPipelineHelpers,
):
    _configure_supernet_retrain_replay(args, task_id)

    retrain_path = os.path.join(current_task_path, "learned_net")
    if args.skip_search and not os.path.exists(os.path.join(retrain_path, "net.config")):
        os.makedirs(retrain_path, exist_ok=True)
        global_server.load_model()
        global_server.get_normal_net()
    if os.path.isfile(os.path.join(retrain_path, "net.config")):
        auto_resume_manager.handle_event(
            task_id,
            "learned_net_ready",
            learned_net_path=retrain_path,
        )
    args.path = retrain_path
    os.makedirs(args.path, exist_ok=True)

    args.search = False
    retrain_start_round = args.start_round if args.retrain_start_round is None else args.retrain_start_round
    retrain_last_round = args.last_round if args.retrain_last_round is None else args.retrain_last_round
    retrain_run_config_kwargs = _build_retrain_run_config_kwargs(args, retrain_last_round)
    retrain_resume_plan = (
        runtime_context.resume_task_plan
        if isinstance(runtime_context.resume_task_plan, dict)
        and runtime_context.resume_task_plan.get("phase") == "retrain"
        else None
    )
    resume_stage_start_index = int(retrain_resume_plan.get("retrain_stage_start_index", 0)) if retrain_resume_plan else 0
    resume_running_stage = bool(retrain_resume_plan.get("retrain_resume_running_stage", False)) if retrain_resume_plan else False
    retrain_bootstrap_checkpoint_path = retrain_resume_plan.get("retrain_bootstrap_checkpoint_path") if retrain_resume_plan else None
    retrain_teacher_snapshot_path = retrain_resume_plan.get("retrain_teacher_snapshot_path") if retrain_resume_plan else None
    args.resume = resume_running_stage if retrain_resume_plan is not None else user_resume

    args.client_id = 0
    global_run_config = CifarRunConfig(**retrain_run_config_kwargs, is_client=False)
    helpers.attach_replay_cfg(global_run_config, args)

    net_config_path = os.path.join(args.path, "net.config")
    net = None
    if os.path.isfile(net_config_path):
        from models import get_net_by_name

        net_config = json.load(open(net_config_path, "r"))
        net = get_net_by_name(net_config["name"]).build_from_config(net_config)
        init_weight_path = os.path.join(args.path, "init")
        if os.path.isfile(init_weight_path):
            try:
                ckpt = torch.load(init_weight_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                missing, unexpected = net.load_state_dict(state_dict, strict=False)
                sig = helpers.model_signature(net)
                print(
                    f"[Retrain] Loaded init weights from {init_weight_path}, "
                    f"missing: {len(missing)}, unexpected: {len(unexpected)}, "
                    f"sig_global_norm={sig.get('global', {}).get('norm'):.4f}"
                )
            except Exception as exc:
                print(f"[Retrain] Failed to load init weights from {init_weight_path}: {exc}")
        else:
            print(f"[Retrain] init weight {init_weight_path} not found, start from random init")
    else:
        print("net_config_path is not file!")

    with open(os.path.join(args.path, f"inherit_check_task{task_id}.log"), "a") as fout:
        fout.write(
            json.dumps(
                {
                    "stage": "retrain_init_net",
                    "task_id": task_id,
                    "signature": helpers.model_signature(net),
                }
            )
            + "\n"
        )

    teacher_model = None
    need_teacher = teacher_is_required(args)
    if need_teacher and task_id > 1:
        prev_retrain_path = os.path.join(prev_task_path, "learned_net")
        teacher_model = copy.deepcopy(net)
        loaded_teacher = helpers.load_prev_task(teacher_model, prev_retrain_path)
        if not loaded_teacher:
            teacher_model = None
            print(f"[Retrain] 未能从 {prev_retrain_path} 加载教师模型，KD / KD-based ortho 将跳过")
    if need_teacher and retrain_teacher_snapshot_path and resume_stage_start_index > 0:
        if os.path.isdir(retrain_teacher_snapshot_path):
            resumed_teachers = []
            for idx in range(args.num_users):
                snapshot_path = os.path.join(retrain_teacher_snapshot_path, f"client_{idx}.pth")
                resumed_teacher = helpers.load_teacher_snapshot_like(
                    net,
                    snapshot_path,
                    desc=f"AutoResumeTeacherClient{idx}",
                )
                resumed_teachers.append(resumed_teacher)
            if any(t is not None for t in resumed_teachers):
                teacher_model = resumed_teachers
                print(f"[AutoResume] 用 per-client stage 快照恢复 teacher: {retrain_teacher_snapshot_path}")
        else:
            resumed_teacher = helpers.load_teacher_snapshot_like(
                net,
                retrain_teacher_snapshot_path,
                desc="AutoResumeTeacher",
            )
            if resumed_teacher is not None:
                teacher_model = resumed_teacher
                print(f"[AutoResume] 用 stage 快照恢复 teacher: {retrain_teacher_snapshot_path}")

    global_run_manager = RunManager(
        args.path,
        copy.deepcopy(net),
        global_run_config,
        init_model=False,
        task_id=task_id,
        run_phase="personalized_subnet_retrain",
    )
    global_run_manager.save_config(print_info=True)
    if teacher_model is not None:
        global_run_manager.set_teacher(teacher_model)

    base_retrain_state = copy.deepcopy(global_run_manager.net.module.state_dict())
    clients, all_client_idx_arr = [], []
    client_path_root = os.path.join(args.path, "clients")
    os.makedirs(client_path_root, exist_ok=True)
    client_subnet_dir = os.path.join(current_task_path, "client_subnets")
    for idx in range(args.num_users):
        all_client_idx_arr.append(idx)
        client_artifact_path = os.path.join(client_subnet_dir, f"client_{idx}_task_{task_id}_subnet.pt")
        client_net, _ = _load_client_subnet_artifact(client_artifact_path)
        subnet_source = "client_subnet"
        if client_net is None:
            client_net = copy.deepcopy(global_run_manager.net.module)
            client_net.load_state_dict(base_retrain_state, strict=False)
            client_artifact_path = None
            subnet_source = "learned_net_fallback"
        client_path = os.path.join(client_path_root, f"client_{idx}")
        os.makedirs(client_path, exist_ok=True)
        client = RunManager(
            client_path,
            client_net,
            _prepare_retrain_run_config(
                clients_run_config_arr[idx],
                retrain_last_round,
                local_epoch_number=getattr(args, "local_epoch_number", 1),
            ),
            init_model=False,
            task_id=task_id,
            replay_buffer=replay_buffers_across_tasks[idx],
            run_phase="personalized_subnet_retrain",
        )
        helpers.attach_replay_cfg(client.run_config, args)
        client.run_config.search = False
        client.preserve_local_model_for_first_sync = client_artifact_path is not None
        client.personalized_subnet_artifact = client_artifact_path
        client.personalized_subnet_source = subnet_source
        client.save_config(print_info=False)
        atomic_torch_save(
            {"state_dict": client.net.module.state_dict(), "dataset": client.run_config.dataset},
            os.path.join(client.path, "init"),
        )
        clients.append(client)
        if teacher_model is not None:
            client.set_teacher(teacher_model)
        print(
            "The {} user has {} training data and {} test data.".format(
                idx,
                client.run_config.data_provider.trn_set_length,
                client.run_config.data_provider.tst_set_length,
            )
        )

    if retrain_bootstrap_checkpoint_path and resume_stage_start_index > 0 and not resume_running_stage:
        helpers.restore_retrain_bootstrap(
            retrain_bootstrap_checkpoint_path,
            global_run_manager,
            clients,
        )

    full_retrain_task_schedule = list(range(1, task_id + 1)) if args.retrain_sequence_from_task1 else [task_id]
    retrain_task_schedule = full_retrain_task_schedule[resume_stage_start_index:]
    args.skip_retrain_log_cleanup = False
    def _set_teacher_all(teacher):
        if isinstance(teacher, list):
            first_teacher = next((t for t in teacher if t is not None), None)
            global_run_manager.set_teacher(first_teacher)
            for idx, rm in enumerate(clients):
                chosen_teacher = teacher[idx] if idx < len(teacher) else first_teacher
                rm.set_teacher(chosen_teacher)
            return
        global_run_manager.set_teacher(teacher)
        for rm in clients:
            rm.set_teacher(teacher)

    if teacher_model is not None:
        _set_teacher_all(teacher_model)

    if args.enable_replay and retrain_resume_plan is None and not args.retrain_sequence_from_task1 and task_id > 1:
        print(f"[ReplayPrefill] retrain_sequence_from_task1=OFF，预先填充任务 1~{task_id - 1} 的样本到 replay buffer")
        tasks_to_fill = list(range(1, task_id))
        _rebuild_replay_buffer(global_run_manager, tasks_to_fill)
        for client_rm in clients:
            _rebuild_replay_buffer(client_rm, tasks_to_fill)
        _reset_run_manager_task(global_run_manager, task_id)
        for client_rm in clients:
            _reset_run_manager_task(client_rm, task_id)
    elif args.enable_replay and retrain_resume_plan is None and args.retrain_sequence_from_task1 and task_id > 1:
        print(f"[ReplayPrefill] retrain_sequence_from_task1=ON，构建 1~{task_id - 1} 混合样本用于随机重放")
        tasks_to_fill = list(range(1, task_id))
        _rebuild_replay_buffer(global_run_manager, tasks_to_fill)
        for client_rm in clients:
            _rebuild_replay_buffer(client_rm, tasks_to_fill)

    auto_resume_manager.handle_event(
        task_id,
        "retrain_started",
        mode="supernet",
        retrain_path=args.path,
        stage_index=resume_stage_start_index,
        stage_task_id=retrain_task_schedule[0] if retrain_task_schedule else None,
        stage_status="running" if resume_running_stage else "pending",
    )

    teacher_snapshot_path = None
    for stage_idx, retrain_task_id in enumerate(retrain_task_schedule, start=resume_stage_start_index):
        print(f"[Retrain] 开始顺序重训 task {retrain_task_id}/{task_id}")
        if stage_idx > 0:
            args.skip_retrain_log_cleanup = True
        replay_only_stage = args.enable_replay and args.retrain_sequence_from_task1 and retrain_task_id < task_id
        auto_resume_manager.handle_event(
            task_id,
            "retrain_stage_started",
            stage_index=stage_idx,
            stage_task_id=retrain_task_id,
        )
        _reset_run_manager_task(global_run_manager, retrain_task_id)
        if args.retrain_sequence_from_task1:
            global_run_manager.reset_forgetting_stats()
        if args.enable_replay and args.retrain_sequence_from_task1 and retrain_task_id < task_id:
            global_run_manager.stage_training_buffer = _build_stage_training_buffer(global_run_manager, retrain_task_id)
            global_run_manager.allow_mix_during_stage = True
            for client_rm in clients:
                client_rm.stage_training_buffer = _build_stage_training_buffer(client_rm, retrain_task_id)
                client_rm.allow_mix_during_stage = True
                _reset_run_manager_task(client_rm, retrain_task_id)
                if args.retrain_sequence_from_task1:
                    client_rm.reset_forgetting_stats()
            print(f"[ReplayPrefill] task{retrain_task_id} stage buffers ready")
        else:
            for client_rm in clients:
                _reset_run_manager_task(client_rm, retrain_task_id)
                if args.retrain_sequence_from_task1:
                    client_rm.reset_forgetting_stats()
            global_run_manager.stage_training_buffer = None
            global_run_manager.allow_mix_during_stage = False
            for client_rm in clients:
                client_rm.stage_training_buffer = None
                client_rm.allow_mix_during_stage = False
        global_run_manager.replay_only_training = replay_only_stage
        for client_rm in clients:
            client_rm.replay_only_training = replay_only_stage
        if teacher_model is not None:
            _set_teacher_all(teacher_model)

        def _retrain_progress_callback(event, **payload):
            io_time = 0.0
            if args.enable_replay:
                for idx, rm in enumerate(clients):
                    replay_buffers_across_tasks[idx] = rm.replay_buffer
                io_time += helpers.save_replay_buffers(current_task_path)

            auto_resume_manager.handle_event(
                task_id,
                "retrain_round_completed",
                round_idx=payload.get("round_idx"),
                checkpoint_path=payload.get("checkpoint_path"),
                stage_index=stage_idx,
                stage_task_id=retrain_task_id,
                stage_status="running",
                replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
            )
            return io_time

        runtime_context.set_progress_callback(_retrain_progress_callback)
        helpers.train_personalized_subnet(
            args,
            global_run_manager,
            clients,
            all_client_idx_arr,
            start_round=retrain_start_round,
            last_round=retrain_last_round,
        )
        runtime_context.clear_progress_callback()
        stage_checkpoint_io_time = 0.0
        teacher_snapshot_path = os.path.join(args.path, "teacher_snapshots", f"task{retrain_task_id}")
        if teacher_is_required(args):
            os.makedirs(teacher_snapshot_path, exist_ok=True)
            for idx, rm in enumerate(clients):
                try:
                    io_start = time.time()
                    atomic_torch_save({"state_dict": rm.net.module.state_dict()}, os.path.join(teacher_snapshot_path, f"client_{idx}.pth"))
                    stage_checkpoint_io_time += time.time() - io_start
                except Exception as exc:
                    print(f"[Retrain] Failed to save teacher snapshot for client {idx}: {exc}")
            print(f"[Retrain] Saved per-client teacher snapshots to {teacher_snapshot_path}")
        else:
            teacher_snapshot_path = None

        if stage_idx < len(full_retrain_task_schedule) - 1 and teacher_is_required(args):
            teacher_model = [copy.deepcopy(rm.net.module) for rm in clients]
            _set_teacher_all(teacher_model)
        if args.enable_replay:
            for idx, rm in enumerate(clients):
                replay_buffers_across_tasks[idx] = rm.replay_buffer
            stage_checkpoint_io_time += helpers.save_replay_buffers(current_task_path)

        next_stage_index = stage_idx + 1
        next_stage_task_id = full_retrain_task_schedule[next_stage_index] if next_stage_index < len(full_retrain_task_schedule) else None
        auto_resume_manager.handle_event(
            task_id,
            "retrain_stage_completed",
            stage_index=stage_idx,
            stage_task_id=retrain_task_id,
            next_stage_index=next_stage_index,
            next_stage_task_id=next_stage_task_id,
            next_stage_status="pending" if next_stage_task_id is not None else "completed",
            last_round=retrain_last_round - 1,
            checkpoint_path=os.path.join(args.path, "clients"),
            teacher_snapshot_path=teacher_snapshot_path,
            replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
            completed_stage_ids=full_retrain_task_schedule[:next_stage_index],
        )
        print(f"[Retrain] stage task{retrain_task_id} finalize checkpoint_io={stage_checkpoint_io_time / 60:.4f}m")
        print(f"[Retrain] task {retrain_task_id} 重训完成")
        global_run_manager.replay_only_training = False
        for client_rm in clients:
            client_rm.replay_only_training = False
        global_run_manager.stage_training_buffer = None
        global_run_manager.allow_mix_during_stage = False
        for client_rm in clients:
            client_rm.stage_training_buffer = None
            client_rm.allow_mix_during_stage = False

    print("所有客户端重训完成")
    if args.enable_replay:
        for idx, rm in enumerate(clients):
            replay_buffers_across_tasks[idx] = rm.replay_buffer
    helpers.save_replay_buffers(current_task_path)
    auto_resume_manager.handle_event(
        task_id,
        "retrain_completed",
        checkpoint_path=os.path.join(args.path, "clients"),
        teacher_snapshot_path=teacher_snapshot_path,
        replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
    )


def run_baseline_retrain_pipeline(
    args,
    task_id: int,
    prev_task_path: str,
    current_task_path: str,
    replay_buffers_across_tasks,
    runtime_context,
    auto_resume_manager,
    helpers: RetrainPipelineHelpers,
):
    baseline_method = _configure_baseline_method(args)
    print(
        f"[Baseline] method={baseline_method}, retrain_fedavg={args.retrain_fedavg}, "
        f"enable_replay={getattr(args, 'enable_replay', False)}, "
        f"enable_kd={getattr(args, 'enable_kd', False)}"
    )
    if baseline_method == "fedta":
        print(
            "[Baseline] FedTA config: "
            f"tail_ratio={getattr(args, 'fedta_tail_ratio', 0.4)}, "
            f"anchor_lambda={getattr(args, 'fedta_anchor_lambda', 0.5)}, "
            f"temperature={getattr(args, 'fedta_temperature', 2.0)}, "
            f"min_tail_classes={getattr(args, 'fedta_min_tail_classes', 1)}"
        )
    elif baseline_method == "af_fcl":
        print(
            "[Baseline] AF-FCL config: "
            f"replay_mode={getattr(args, 'replay_mode', 'none')}, "
            f"replay_capacity={getattr(args, 'replay_capacity', 0)}, "
            f"replay_capacity_ratio={getattr(args, 'replay_capacity_ratio', None)}, "
            f"replay_per_batch={getattr(args, 'replay_per_batch', 0)}, "
            f"replay_old_task_scale={getattr(args, 'replay_old_task_scale', 1.0)}, "
            f"replay_old_task_scale_by_F={getattr(args, 'replay_old_task_scale_by_F', 0.0)}, "
            f"kd_method={getattr(args, 'cl_kd_method', 'none')}, "
            f"kd_lambda={getattr(args, 'cl_kd_logit_lambda', 0.0)}, "
            f"kd_temperature={getattr(args, 'cl_kd_temperature', 1.0)}"
        )
    args.search = False
    retrain_last_round = args.last_round if args.retrain_last_round is None else args.retrain_last_round
    retrain_run_config_kwargs = _build_retrain_run_config_kwargs(args, retrain_last_round)
    args.client_id = 0
    global_run_config = CifarRunConfig(**retrain_run_config_kwargs, is_client=False)
    helpers.attach_replay_cfg(global_run_config, args)
    global_net = BaselineResNet(
        arch=args.baseline_arch,
        num_classes=global_run_config.data_provider.n_classes,
        pretrained=args.baseline_pretrained,
    )

    teacher_model = None
    loaded_prev = False
    if task_id > 1:
        try:
            loaded_prev = helpers.load_prev_task(global_net, prev_task_path)
        except Exception as exc:
            print(f"[Baseline] 尝试从 {prev_task_path} 加载上一任务权重失败: {exc}")
            loaded_prev = False

    if not loaded_prev:
        print(f"[Baseline] 未找到上一任务权重，task {task_id} 从随机初始化开始")
        global_net.init_model(args.model_init, args.init_div_groups)
    else:
        print(f"[Baseline] 已从 {prev_task_path} 继承权重，task {task_id} 在上一任务模型上继续训练")
        if teacher_is_required(args):
            teacher_model = BaselineResNet(
                arch=args.baseline_arch,
                num_classes=global_run_config.data_provider.n_classes,
                pretrained=args.baseline_pretrained,
            )
            teacher_model.load_state_dict(copy.deepcopy(global_net.state_dict()), strict=False)
            print(f"[Baseline] 已从 {prev_task_path} 继承教师模型")

    base_fixed_state = copy.deepcopy(global_net.state_dict())
    global_run_manager = RunManager(
        args.path,
        global_net,
        global_run_config,
        init_model=False,
        task_id=task_id,
        run_phase="personalized_subnet_retrain",
    )
    global_run_manager.save_config(print_info=True)
    if teacher_model is not None:
        global_run_manager.set_teacher(teacher_model)

    clients, all_client_idx_arr = [], []
    client_path_root = os.path.join(args.path, "clients")
    os.makedirs(client_path_root, exist_ok=True)
    for idx in range(args.num_users):
        args.client_id = idx
        client_run_config_kwargs = dict(retrain_run_config_kwargs)
        client_run_config_kwargs["client_id"] = idx
        client_run_config = CifarRunConfig(**client_run_config_kwargs, is_client=True)
        helpers.attach_replay_cfg(client_run_config, args)
        _prepare_retrain_run_config(
            client_run_config,
            retrain_last_round,
            local_epoch_number=getattr(args, "local_epoch_number", 1),
        )
        client_net = BaselineResNet(
            arch=args.baseline_arch,
            num_classes=client_run_config.data_provider.n_classes,
            pretrained=args.baseline_pretrained,
        )
        client_net.load_state_dict(base_fixed_state, strict=False)
        client = RunManager(
            os.path.join(client_path_root, f"client_{idx}"),
            client_net,
            client_run_config,
            init_model=False,
            task_id=task_id,
            replay_buffer=replay_buffers_across_tasks[idx],
            run_phase="personalized_subnet_retrain",
        )
        client.save_config(print_info=False)
        atomic_torch_save(
            {"state_dict": client.net.module.state_dict(), "dataset": client.run_config.dataset},
            os.path.join(client.path, "init"),
        )
        clients.append(client)
        all_client_idx_arr.append(idx)
        print(
            "The {} user has {} training data and {} test data.".format(
                idx,
                client.run_config.data_provider.trn_set_length,
                client.run_config.data_provider.tst_set_length,
            )
        )
        if teacher_model is not None:
            client.set_teacher(teacher_model)

    all_clients = CommonwealthMachine(
        target_hardware=args.baseline_arch,
        config=args,
        global_run_manager=global_run_manager,
        clients_idx_arr=all_client_idx_arr,
        clients=clients,
        start_round=args.retrain_start_round,
        last_round=retrain_last_round,
        path=args.path,
    )

    auto_resume_manager.handle_event(
        task_id,
        "retrain_started",
        mode="baseline",
        retrain_path=args.path,
        stage_index=0,
        stage_task_id=task_id,
        stage_status="running" if args.resume else "pending",
    )

    def _baseline_progress_callback(event, **payload):
        io_time = 0.0
        if args.enable_replay:
            for idx, rm in enumerate(clients):
                replay_buffers_across_tasks[idx] = rm.replay_buffer
            io_time += helpers.save_replay_buffers(current_task_path)
        auto_resume_manager.handle_event(
            task_id,
            "retrain_round_completed",
            round_idx=payload.get("round_idx"),
            checkpoint_path=payload.get("checkpoint_path"),
            stage_index=0,
            stage_task_id=task_id,
            stage_status="running",
            replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
        )
        return io_time

    runtime_context.set_progress_callback(_baseline_progress_callback)
    all_clients.run()
    runtime_context.clear_progress_callback()

    for idx, rm in enumerate(clients):
        replay_buffers_across_tasks[idx] = rm.replay_buffer
    baseline_finalize_io_time = 0.0
    if args.enable_replay:
        baseline_finalize_io_time += helpers.save_replay_buffers(args.path)

    print(f"[Baseline] finalize checkpoint_io={baseline_finalize_io_time / 60:.4f}m")
    auto_resume_manager.handle_event(
        task_id,
        "retrain_completed",
        checkpoint_path=os.path.join(args.path, "clients"),
        replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
    )
