import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List

import torch

from auto_resume import atomic_torch_save
from commonwealth_machine import CommonwealthMachine
from models.baseline_nets import BaselineResNet
from run_manager import CifarRunConfig, RunManager, SimpleReplayBuffer


@dataclass
class RetrainPipelineHelpers:
    attach_replay_cfg: Callable
    load_prev_task: Callable
    load_state_with_fallback: Callable
    save_state_safely: Callable
    save_replay_buffers: Callable
    load_model_from_checkpoint: Callable
    restore_retrain_bootstrap: Callable
    load_teacher_snapshot_like: Callable
    model_signature: Callable
    train_personalized_subnet: Callable


KD_ORTHO_METHODS = {"kd_ortho", "kd_prev_grad_ortho"}


def teacher_is_required(args) -> bool:
    kd_enabled = bool(getattr(args, "enable_kd", False))
    kd_weight = float(getattr(args, "cl_kd_logit_lambda", 0.0))
    kd_method = str(getattr(args, "cl_kd_method", "none")).lower()
    ortho_enabled = bool(getattr(args, "enable_orthogonal_update", False))
    ortho_method = str(getattr(args, "cl_ortho_method", "none")).lower()
    use_kd_loss = kd_enabled and kd_weight > 0 and kd_method in {"logit", "logit_conf"}
    use_kd_ortho = ortho_enabled and ortho_method in KD_ORTHO_METHODS
    return use_kd_loss or use_kd_ortho


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
    global_run_config = CifarRunConfig(**args.__dict__, is_client=False)
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
    for idx in range(args.num_users):
        all_client_idx_arr.append(idx)
        client_net = copy.deepcopy(global_run_manager.net.module)
        client_net.load_state_dict(base_retrain_state, strict=False)
        client = RunManager(
            args.path,
            client_net,
            clients_run_config_arr[idx],
            init_model=False,
            task_id=task_id,
            replay_buffer=replay_buffers_across_tasks[idx],
            run_phase="personalized_subnet_retrain",
        )
        helpers.attach_replay_cfg(client.run_config, args)
        client.run_config.search = False
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
    ewc_state = None
    ortho_state = None

    def _broadcast_ewc_state(state):
        global_run_manager.load_ewc_state(state)
        for rm in clients:
            rm.load_ewc_state(state)

    def _broadcast_ortho_state(state):
        global_run_manager.load_ortho_state(state)
        for rm in clients:
            rm.load_ortho_state(state)

    def _set_teacher_all(teacher):
        global_run_manager.set_teacher(teacher)
        for rm in clients:
            rm.set_teacher(teacher)

    ewc_state_path = os.path.join(args.path, "ewc_state.pt")
    prev_ewc_state_path = os.path.join(prev_task_path, "learned_net", "ewc_state.pt")
    ortho_state_path = os.path.join(args.path, "ortho_state.pt")
    prev_ortho_state_path = os.path.join(prev_task_path, "learned_net", "ortho_state.pt")
    if args.enable_ewc:
        ewc_state = helpers.load_state_with_fallback(ewc_state_path, prev_ewc_state_path, desc="Retrain")
    if args.enable_orthogonal_update:
        ortho_state = helpers.load_state_with_fallback(ortho_state_path, prev_ortho_state_path, desc="Retrain")

    _broadcast_ewc_state(ewc_state)
    _broadcast_ortho_state(ortho_state)
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
        _broadcast_ewc_state(ewc_state)
        _broadcast_ortho_state(ortho_state)
        if teacher_model is not None:
            _set_teacher_all(teacher_model)

        def _retrain_progress_callback(event, **payload):
            io_time = 0.0
            if args.enable_replay:
                for idx, rm in enumerate(clients):
                    replay_buffers_across_tasks[idx] = rm.replay_buffer
                io_time += helpers.save_replay_buffers(current_task_path)

            current_ewc_state = global_run_manager.export_ewc_state() if args.enable_ewc else None
            if current_ewc_state is not None:
                io_time += helpers.save_state_safely(current_ewc_state, ewc_state_path, desc="RetrainRound")

            current_ortho_state = global_run_manager.export_ortho_state() if args.enable_orthogonal_update else None
            if current_ortho_state is not None:
                io_time += helpers.save_state_safely(current_ortho_state, ortho_state_path, desc="RetrainRound")

            auto_resume_manager.handle_event(
                task_id,
                "retrain_round_completed",
                round_idx=payload.get("round_idx"),
                checkpoint_path=payload.get("checkpoint_path"),
                stage_index=stage_idx,
                stage_task_id=retrain_task_id,
                stage_status="running",
                replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
                ewc_state_path=ewc_state_path if os.path.isfile(ewc_state_path) else None,
                ortho_state_path=ortho_state_path if os.path.isfile(ortho_state_path) else None,
            )
            return io_time

        runtime_context.set_progress_callback(_retrain_progress_callback)
        global_run_manager = helpers.train_personalized_subnet(
            args,
            global_run_manager,
            clients,
            all_client_idx_arr,
            start_round=retrain_start_round,
            last_round=retrain_last_round,
        )
        runtime_context.clear_progress_callback()
        stage_checkpoint_io_time = 0.0
        teacher_snapshot_path = os.path.join(args.path, f"teacher_task{retrain_task_id}.pth")
        try:
            io_start = time.time()
            atomic_torch_save({"state_dict": global_run_manager.net.module.state_dict()}, teacher_snapshot_path)
            stage_checkpoint_io_time += time.time() - io_start
            print(f"[Retrain] Saved teacher snapshot to {teacher_snapshot_path}")
        except Exception as exc:
            print(f"[Retrain] Failed to save teacher snapshot: {exc}")

        if stage_idx < len(full_retrain_task_schedule) - 1:
            teacher_model = copy.deepcopy(global_run_manager.net.module)
            _set_teacher_all(teacher_model)
        if args.enable_ewc and args.ewc_lambda > 0:
            fisher, _ = global_run_manager.compute_importance(max_samples=args.ewc_samples_per_task)
            if fisher is not None:
                global_run_manager.consolidate_ewc(fisher)
                ewc_state = global_run_manager.export_ewc_state()
                stage_checkpoint_io_time += helpers.save_state_safely(ewc_state, ewc_state_path, desc="Retrain")
        if args.enable_orthogonal_update and args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
            ortho_ref, _ = global_run_manager.compute_ortho_reference(max_samples=args.ortho_samples_per_task)
            if ortho_ref is not None:
                ortho_state = global_run_manager.export_ortho_state()
                stage_checkpoint_io_time += helpers.save_state_safely(ortho_state, ortho_state_path, desc="Retrain")
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
            checkpoint_path=os.path.join(args.path, "checkpoint", "checkpoint.pth.tar"),
            teacher_snapshot_path=teacher_snapshot_path,
            ewc_state_path=ewc_state_path if os.path.isfile(ewc_state_path) else None,
            ortho_state_path=ortho_state_path if os.path.isfile(ortho_state_path) else None,
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
        checkpoint_path=os.path.join(args.path, "checkpoint", "checkpoint.pth.tar"),
        teacher_snapshot_path=teacher_snapshot_path,
        ewc_state_path=ewc_state_path if os.path.isfile(ewc_state_path) else None,
        ortho_state_path=ortho_state_path if os.path.isfile(ortho_state_path) else None,
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
    args.search = False
    args.client_id = 0
    global_run_config = CifarRunConfig(**args.__dict__, is_client=False)
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
    for idx in range(args.num_users):
        args.client_id = idx
        client_run_config = CifarRunConfig(**args.__dict__, is_client=True)
        helpers.attach_replay_cfg(client_run_config, args)
        client_run_config.search = False
        client_net = BaselineResNet(
            arch=args.baseline_arch,
            num_classes=client_run_config.data_provider.n_classes,
            pretrained=args.baseline_pretrained,
        )
        client_net.load_state_dict(base_fixed_state, strict=False)
        client = RunManager(
            args.path,
            client_net,
            client_run_config,
            init_model=False,
            task_id=task_id,
            replay_buffer=replay_buffers_across_tasks[idx],
            run_phase="personalized_subnet_retrain",
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
        last_round=args.retrain_last_round,
        path=args.path,
    )

    ewc_state_path = os.path.join(args.path, "ewc_state.pt")
    prev_ewc_state_path = os.path.join(prev_task_path, "ewc_state.pt")
    ewc_state = None
    if args.enable_ewc and os.path.isfile(ewc_state_path):
        try:
            ewc_state = torch.load(ewc_state_path, map_location="cpu")
            print(f"[Baseline] Loaded EWC state from {ewc_state_path}")
        except Exception as exc:
            print(f"[Baseline] Failed to load EWC state: {exc}")
    elif args.enable_ewc and os.path.isfile(prev_ewc_state_path):
        try:
            ewc_state = torch.load(prev_ewc_state_path, map_location="cpu")
            print(f"[Baseline] Loaded EWC state from prev task {prev_ewc_state_path}")
        except Exception as exc:
            print(f"[Baseline] Failed to load prev-task EWC state: {exc}")

    def _broadcast_ewc_state(state):
        global_run_manager.load_ewc_state(state)
        for rm in clients:
            rm.load_ewc_state(state)

    _broadcast_ewc_state(ewc_state)

    ortho_state = None
    current_ortho_state_path = os.path.join(args.path, "ortho_state.pt")
    prev_ortho_state_path = os.path.join(prev_task_path, "ortho_state.pt")
    if args.enable_orthogonal_update:
        ortho_state = helpers.load_state_with_fallback(
            current_ortho_state_path,
            prev_ortho_state_path,
            desc="Baseline",
        )

    global_run_manager.load_ortho_state(ortho_state)
    for client in clients:
        client.load_ortho_state(ortho_state)

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
        current_ewc_state = global_run_manager.export_ewc_state() if args.enable_ewc else None
        if current_ewc_state is not None:
            io_time += helpers.save_state_safely(current_ewc_state, ewc_state_path, desc="BaselineRound")
        current_ortho_state = global_run_manager.export_ortho_state() if args.enable_orthogonal_update else None
        if current_ortho_state is not None:
            io_time += helpers.save_state_safely(current_ortho_state, current_ortho_state_path, desc="BaselineRound")
        auto_resume_manager.handle_event(
            task_id,
            "retrain_round_completed",
            round_idx=payload.get("round_idx"),
            checkpoint_path=payload.get("checkpoint_path"),
            stage_index=0,
            stage_task_id=task_id,
            stage_status="running",
            replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
            ewc_state_path=ewc_state_path if os.path.isfile(ewc_state_path) else None,
            ortho_state_path=current_ortho_state_path if os.path.isfile(current_ortho_state_path) else None,
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

    if args.enable_ewc and args.ewc_lambda > 0:
        print(f"[Baseline] 计算并整合 Fisher 信息，lambda={args.ewc_lambda}")
        fisher, processed = global_run_manager.compute_importance(max_samples=args.ewc_samples_per_task)
        if fisher is None:
            print(f"[Baseline] Fisher/importance is None (processed={processed}), skip consolidate")
        else:
            global_run_manager.consolidate_ewc(fisher, update_prev_params=True)
            ewc_state = global_run_manager.export_ewc_state()
            print(
                f"[Baseline] Importance_keys={len(fisher)}, "
                f"importance_norm={sum(v.sum().item() for v in fisher.values()):.4f}, processed={processed}"
            )
            try:
                io_start = time.time()
                atomic_torch_save(ewc_state, ewc_state_path)
                baseline_finalize_io_time += time.time() - io_start
                print(f"[Baseline] Saved EWC state to {ewc_state_path}")
            except Exception as exc:
                print(f"[Baseline] Failed to save EWC state: {exc}")

    if args.enable_orthogonal_update and args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
        ortho_ref, processed = global_run_manager.compute_ortho_reference(max_samples=args.ortho_samples_per_task)
        if ortho_ref is not None and isinstance(ortho_ref, dict):
            ref_norm = ortho_ref["global"].norm() if "global" in ortho_ref else sum(v.norm() for v in ortho_ref.values())
            print(f"[Baseline] Ortho ref norm={ref_norm:.4f}, processed={processed}")
            ortho_state = global_run_manager.export_ortho_state()
            try:
                io_start = time.time()
                atomic_torch_save(ortho_state, current_ortho_state_path)
                baseline_finalize_io_time += time.time() - io_start
                print(f"[Baseline] Saved ortho state to {current_ortho_state_path}")
            except Exception as exc:
                print(f"[Baseline] Failed to save ortho state: {exc}")
        else:
            print(f"[Baseline] Ortho ref is None (processed={processed}), skip saving ortho_state")

    print(f"[Baseline] finalize checkpoint_io={baseline_finalize_io_time / 60:.4f}m")
    auto_resume_manager.handle_event(
        task_id,
        "retrain_completed",
        checkpoint_path=os.path.join(args.path, "checkpoint", "checkpoint.pth.tar"),
        ewc_state_path=ewc_state_path if os.path.isfile(ewc_state_path) else None,
        ortho_state_path=current_ortho_state_path if os.path.isfile(current_ortho_state_path) else None,
        replay_buffer_path=os.path.join(current_task_path, "replay_buffers.pt") if args.enable_replay else None,
    )
