import copy
import json
import os
import tempfile
import time

import torch


def atomic_json_dump(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=True, indent=2, sort_keys=True)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_torch_save(state, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(path))
    fd, tmp_path = tempfile.mkstemp(prefix=f".tmp_{base}_", suffix=ext or ".pt", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _deep_update(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class AutoResumeManager:
    def __init__(self, base_task_path: str):
        self.base_task_path = base_task_path

    def task_dir(self, task_id: int) -> str:
        return f"{self.base_task_path}-task{task_id}"

    def state_path(self, task_id: int) -> str:
        return os.path.join(self.task_dir(task_id), "auto_resume_state.json")

    def _default_state(self, task_id: int, object_to_search: str = None) -> dict:
        return {
            "version": 1,
            "task_id": int(task_id),
            "object_to_search": object_to_search,
            "status": "running",
            "phase": "init",
            "task_completed": False,
            "updated_at": None,
            "search": {
                "warmup_status": "pending",
                "warmup_last_round": None,
                "warmup_checkpoint": None,
                "search_status": "pending",
                "search_last_round": None,
                "search_checkpoint": None,
            },
            "artifacts": {
                "learned_net_ready": False,
                "learned_net_path": None,
                "replay_buffer_path": None,
            },
            "retrain": {
                "mode": None,
                "status": "pending",
                "path": None,
                "last_round": None,
                "checkpoint_path": None,
                "current_stage_index": 0,
                "current_stage_task_id": None,
                "current_stage_status": "pending",
                "completed_stage_ids": [],
                "bootstrap_checkpoint_path": None,
                "teacher_snapshot_path": None,
                "ewc_state_path": None,
                "ortho_state_path": None,
            },
        }

    def load_task_state(self, task_id: int):
        path = self.state_path(task_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fin:
                return json.load(fin)
        except Exception:
            return None

    def update_task_state(self, task_id: int, updates: dict, object_to_search: str = None):
        state = self.load_task_state(task_id)
        if state is None:
            state = self._default_state(task_id, object_to_search=object_to_search)
        elif object_to_search is not None and not state.get("object_to_search"):
            state["object_to_search"] = object_to_search
        _deep_update(state, updates)
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        atomic_json_dump(state, self.state_path(task_id))
        return state

    def handle_event(self, task_id: int, event: str, **payload):
        if event == "task_started":
            existing = self.load_task_state(task_id)
            if existing is not None and not existing.get("task_completed"):
                return self.update_task_state(
                    task_id,
                    {
                        "status": "running",
                        "object_to_search": payload.get("object_to_search") or existing.get("object_to_search"),
                    },
                )
            return self.update_task_state(
                task_id,
                {
                    "status": "running",
                    "phase": "init",
                    "task_completed": False,
                },
                object_to_search=payload.get("object_to_search"),
            )

        if event == "warmup_started":
            return self.update_task_state(
                task_id,
                {
                    "phase": "warmup",
                    "search": {
                        "warmup_status": "running",
                    },
                },
            )

        if event == "warmup_round_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "warmup",
                    "search": {
                        "warmup_status": "running",
                        "warmup_last_round": int(payload.get("round_idx", 0)),
                        "warmup_checkpoint": payload.get("checkpoint_path"),
                    },
                },
            )

        if event == "warmup_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "search",
                    "search": {
                        "warmup_status": "completed",
                        "warmup_checkpoint": payload.get("checkpoint_path"),
                        "search_status": "pending",
                    },
                },
            )

        if event == "search_started":
            return self.update_task_state(
                task_id,
                {
                    "phase": "search",
                    "search": {
                        "search_status": "running",
                    },
                },
            )

        if event == "search_round_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "search",
                    "search": {
                        "warmup_status": "completed",
                        "search_status": "running",
                        "search_last_round": int(payload.get("round_idx", 0)),
                        "search_checkpoint": payload.get("checkpoint_path"),
                    },
                },
            )

        if event == "search_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "search",
                    "search": {
                        "warmup_status": "completed",
                        "search_status": "completed",
                        "search_checkpoint": payload.get("checkpoint_path"),
                    },
                },
            )

        if event == "learned_net_ready":
            return self.update_task_state(
                task_id,
                {
                    "phase": "retrain",
                    "artifacts": {
                        "learned_net_ready": True,
                        "learned_net_path": payload.get("learned_net_path"),
                    },
                    "search": {
                        "warmup_status": "completed",
                        "search_status": "completed",
                    },
                },
            )

        if event == "retrain_started":
            updates = {
                "phase": "retrain",
                "retrain": {
                    "mode": payload.get("mode"),
                    "status": "running",
                    "path": payload.get("retrain_path"),
                    "current_stage_index": int(payload.get("stage_index", 0)),
                    "current_stage_task_id": payload.get("stage_task_id"),
                    "current_stage_status": payload.get("stage_status", "pending"),
                },
            }
            return self.update_task_state(task_id, updates)

        if event == "retrain_stage_started":
            return self.update_task_state(
                task_id,
                {
                    "phase": "retrain",
                    "retrain": {
                        "status": "running",
                        "current_stage_index": int(payload.get("stage_index", 0)),
                        "current_stage_task_id": int(payload.get("stage_task_id")),
                        "current_stage_status": "running",
                    },
                },
            )

        if event == "retrain_round_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "retrain",
                    "retrain": {
                        "status": "running",
                        "last_round": int(payload.get("round_idx", 0)),
                        "checkpoint_path": payload.get("checkpoint_path"),
                        "current_stage_index": int(payload.get("stage_index", 0)),
                        "current_stage_task_id": int(payload.get("stage_task_id"))
                        if payload.get("stage_task_id") is not None
                        else None,
                        "current_stage_status": payload.get("stage_status", "running"),
                        "ewc_state_path": payload.get("ewc_state_path"),
                        "ortho_state_path": payload.get("ortho_state_path"),
                    },
                    "artifacts": {
                        "replay_buffer_path": payload.get("replay_buffer_path"),
                    },
                },
            )

        if event == "retrain_stage_completed":
            completed_stage_ids = payload.get("completed_stage_ids")
            if completed_stage_ids is None:
                completed_stage_ids = [int(payload.get("stage_task_id"))]
            completed_stage_ids = sorted({int(v) for v in completed_stage_ids})
            return self.update_task_state(
                task_id,
                {
                    "phase": "retrain",
                    "retrain": {
                        "status": "running",
                        "last_round": payload.get("last_round"),
                        "checkpoint_path": payload.get("checkpoint_path"),
                        "bootstrap_checkpoint_path": payload.get("checkpoint_path"),
                        "current_stage_index": int(payload.get("next_stage_index", 0)),
                        "current_stage_task_id": payload.get("next_stage_task_id"),
                        "current_stage_status": payload.get("next_stage_status", "completed"),
                        "completed_stage_ids": completed_stage_ids,
                        "teacher_snapshot_path": payload.get("teacher_snapshot_path"),
                        "ewc_state_path": payload.get("ewc_state_path"),
                        "ortho_state_path": payload.get("ortho_state_path"),
                    },
                    "artifacts": {
                        "replay_buffer_path": payload.get("replay_buffer_path"),
                    },
                },
            )

        if event == "retrain_completed":
            return self.update_task_state(
                task_id,
                {
                    "phase": "retrain",
                    "retrain": {
                        "status": "completed",
                        "checkpoint_path": payload.get("checkpoint_path"),
                        "bootstrap_checkpoint_path": payload.get("checkpoint_path"),
                        "current_stage_status": "completed",
                        "teacher_snapshot_path": payload.get("teacher_snapshot_path"),
                        "ewc_state_path": payload.get("ewc_state_path"),
                        "ortho_state_path": payload.get("ortho_state_path"),
                    },
                    "artifacts": {
                        "replay_buffer_path": payload.get("replay_buffer_path"),
                    },
                },
            )

        if event == "task_completed":
            return self.update_task_state(
                task_id,
                {
                    "status": "completed",
                    "phase": "completed",
                    "task_completed": True,
                },
            )
        return None

    def _infer_legacy_state(self, task_id: int, object_to_search: str):
        task_dir = self.task_dir(task_id)
        if not os.path.isdir(task_dir):
            return None

        learned_net_path = os.path.join(task_dir, "learned_net")
        search_ckpt_dir = os.path.join(task_dir, "checkpoint")
        retrain_ckpt_dir = os.path.join(learned_net_path, "checkpoint")

        state = self._default_state(task_id, object_to_search=object_to_search)

        if os.path.isfile(os.path.join(retrain_ckpt_dir, "latest.txt")):
            state["phase"] = "retrain"
            state["search"]["warmup_status"] = "completed"
            state["search"]["search_status"] = "completed"
            state["artifacts"]["learned_net_ready"] = True
            state["artifacts"]["learned_net_path"] = learned_net_path
            state["retrain"]["status"] = "running"
            state["retrain"]["path"] = learned_net_path
            state["retrain"]["checkpoint_path"] = self._read_latest_checkpoint(retrain_ckpt_dir)
            state["retrain"]["current_stage_status"] = "running"
            return state

        if os.path.isfile(os.path.join(learned_net_path, "net.config")):
            state["phase"] = "retrain"
            state["search"]["warmup_status"] = "completed"
            state["search"]["search_status"] = "completed"
            state["artifacts"]["learned_net_ready"] = True
            state["artifacts"]["learned_net_path"] = learned_net_path
            state["retrain"]["status"] = "pending"
            state["retrain"]["path"] = learned_net_path
            return state

        latest_search = self._read_latest_checkpoint(search_ckpt_dir)
        if latest_search is None:
            return None

        state["phase"] = "search"
        if latest_search.endswith("global.pth.tar"):
            state["search"]["warmup_status"] = "completed"
            state["search"]["search_status"] = "running"
            state["search"]["search_checkpoint"] = latest_search
        else:
            try:
                ckpt = torch.load(latest_search, map_location="cpu")
                if ckpt.get("warmup", False):
                    state["phase"] = "warmup"
                    state["search"]["warmup_status"] = "running"
                    state["search"]["warmup_last_round"] = ckpt.get("warmup_round")
                else:
                    state["search"]["warmup_status"] = "completed"
                    state["search"]["search_status"] = "pending"
                state["search"]["warmup_checkpoint"] = latest_search
            except Exception:
                state["phase"] = "warmup"
                state["search"]["warmup_status"] = "running"
                state["search"]["warmup_checkpoint"] = latest_search
        return state

    @staticmethod
    def _read_latest_checkpoint(ckpt_dir: str):
        latest_txt = os.path.join(ckpt_dir, "latest.txt")
        if os.path.isfile(latest_txt):
            try:
                with open(latest_txt, "r", encoding="utf-8") as fin:
                    path = fin.readline().strip()
                if path:
                    return path
            except Exception:
                pass
        for name in ["checkpoint.pth.tar", "global.pth.tar", "warmup.pth.tar"]:
            path = os.path.join(ckpt_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _build_plan_from_state(self, state: dict):
        task_id = int(state["task_id"])
        object_to_search = state.get("object_to_search")
        if state.get("task_completed"):
            return {"task_id": task_id, "completed": True}

        retrain = state.get("retrain", {})
        search = state.get("search", {})
        artifacts = state.get("artifacts", {})

        if object_to_search == "baseline":
            if retrain.get("status") == "running":
                return {
                    "task_id": task_id,
                    "phase": "retrain",
                    "resume": True,
                    "skip_warmup": True,
                    "skip_search": True,
                    "retrain_stage_start_index": 0,
                    "retrain_resume_running_stage": True,
                }
            return {
                "task_id": task_id,
                "phase": "retrain",
                "resume": False,
                "skip_warmup": True,
                "skip_search": True,
                "retrain_stage_start_index": 0,
                "retrain_resume_running_stage": False,
            }

        if retrain.get("status") in {"running", "completed"} or artifacts.get("learned_net_ready"):
            return {
                "task_id": task_id,
                "phase": "retrain",
                "resume": bool(retrain.get("current_stage_status") == "running"),
                "skip_warmup": True,
                "skip_search": True,
                "retrain_stage_start_index": int(retrain.get("current_stage_index") or 0),
                "retrain_resume_running_stage": bool(retrain.get("current_stage_status") == "running"),
                "retrain_bootstrap_checkpoint_path": retrain.get("bootstrap_checkpoint_path"),
                "retrain_teacher_snapshot_path": retrain.get("teacher_snapshot_path"),
                "replay_buffer_path": artifacts.get("replay_buffer_path"),
            }

        if search.get("search_status") == "running":
            return {
                "task_id": task_id,
                "phase": "search",
                "resume": True,
                "skip_warmup": True,
                "skip_search": False,
            }

        if search.get("warmup_status") == "completed":
            return {
                "task_id": task_id,
                "phase": "search",
                "resume": True,
                "skip_warmup": True,
                "skip_search": False,
            }

        if search.get("warmup_status") == "running":
            return {
                "task_id": task_id,
                "phase": "warmup",
                "resume": True,
                "skip_warmup": False,
                "skip_search": False,
            }

        return {
            "task_id": task_id,
            "phase": "init",
            "resume": False,
            "skip_warmup": False,
            "skip_search": False,
        }

    def resolve(self, num_tasks: int, object_to_search: str):
        for task_id in range(1, int(num_tasks) + 1):
            state = self.load_task_state(task_id)
            if state is None:
                state = self._infer_legacy_state(task_id, object_to_search)
            if state is None:
                return {
                    "task_id": task_id,
                    "phase": "init",
                    "resume": False,
                    "skip_warmup": False,
                    "skip_search": False,
                }
            if state.get("task_completed"):
                continue
            return self._build_plan_from_state(state)

        return {
            "all_completed": True,
            "task_id": int(num_tasks) + 1,
        }
