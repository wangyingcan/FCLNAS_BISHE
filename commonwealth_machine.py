import pickle
import time
import warnings
import json

import numpy as np
import torch
from utils_old import save_checkpoint

from nas_manager import ArchSearchRunManager
import time
from utils_old import *
# from models import *
from run_manager import *
from models.super_nets.super_proxyless import *
from tensorboardX import SummaryWriter

class CommonwealthMachine:
    def __init__(self, target_hardware=None, config=None, global_run_manager=None,
                 clients_idx_arr=None, clients=None, start_round=0,
                 last_round=None, path='./'):  # 此处config值得商榷, local_client_list值得商榷
        self.hardware = target_hardware
        self.config = config
        self.global_run_manager = copy.deepcopy(global_run_manager)
        self.clients_idx_arr = clients_idx_arr
        self.clients = clients
        self.start_round = start_round
        self.last_round = last_round
        self.local_epoch_number = config.local_epoch_number
        self.path = path
        self._logs_path, self._save_path = None, None
        if self.hardware is not None:
            self.writerTf = SummaryWriter(comment=self.hardware + 'fed_retrain')
        else:
            self.writerTf = SummaryWriter(comment='fed_retrain')
        print('tensorboardX logdir', self.writerTf.logdir)
        # 非 resume 时清理旧日志，避免跨次运行的 test.log/test_console 累加
        skip_cleanup = getattr(self.config, "skip_retrain_log_cleanup", False)
        if not getattr(self.config, "resume", False) and not skip_cleanup:
            log_dir = os.path.join(self.path, "logs")
            for fname in ["test.log", "test_console.txt", "retrain.log"]:
                fpath = os.path.join(log_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

    def _append_per_client_metric(self, round_idx, client_id, **metrics):
        record = {
            "phase": "retrain",
            "round": int(round_idx),
            "client_id": int(client_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        task_id = getattr(getattr(self.global_run_manager, "run_config", None), "task_id", None)
        if task_id is not None:
            record["task_id"] = int(task_id)
        record.update(metrics)
        metrics_path = os.path.join(self.logs_path, "per_client_metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _append_test_top1_stats(self, round_idx, tag, stats_record):
        record = {
            "phase": "retrain_test_summary",
            "round": int(round_idx),
            "tag": tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        task_id = getattr(getattr(self.global_run_manager, "run_config", None), "task_id", None)
        if task_id is not None:
            record["task_id"] = int(task_id)
        record.update(stats_record)
        stats_path = os.path.join(self.logs_path, "client_test_top1_stats.jsonl")
        with open(stats_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _emit_progress(self, event, **payload):
        runtime_context = getattr(self.config, "runtime_context", None)
        callback = getattr(runtime_context, "progress_callback", None)
        if callable(callback):
            start = time.time()
            try:
                result = callback(event, **payload)
                if isinstance(result, (int, float)):
                    return float(result)
            except Exception as e:
                print(f"[Progress] retrain callback failed for {event}: {e}")
            return max(0.0, time.time() - start)
        return 0.0

    @staticmethod
    def _model_signature_summary(model):
        state = model.state_dict()
        floating = [p.flatten().float() for p in state.values() if torch.is_tensor(p) and p.dtype.is_floating_point]
        if not floating:
            return {"global_norm": 0.0, "global_mean": 0.0, "global_std": 0.0, "classifier_norm": 0.0}
        flat = torch.cat(floating)
        classifier_norm = 0.0
        for key in ["classifier.linear.weight", "classifier.weight", "linear.weight", "fc.weight"]:
            if key in state and torch.is_tensor(state[key]):
                classifier_norm = float(state[key].float().norm())
                break
        return {
            "global_norm": float(flat.norm()),
            "global_mean": float(flat.mean()),
            "global_std": float(flat.std()),
            "classifier_norm": classifier_norm,
        }

    def _log_retrain_init_snapshot(self):
        for idx in self.clients_idx_arr:
            client = self.clients[idx]
            source = getattr(client, "personalized_subnet_source", "unknown")
            artifact = getattr(client, "personalized_subnet_artifact", None)
            sig = self._model_signature_summary(client.net.module)
            try:
                init_loss, init_top1, init_top5 = client.validate(is_test=True, return_top5=True)
            except Exception as e:
                init_loss, init_top1, init_top5 = float("nan"), float("nan"), float("nan")
                self.write_log(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] retrain client{idx} init_eval_failed source={source} error={e}",
                    prefix="retrain",
                )
            self.write_log(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"retrain client{idx} init_eval source={source} "
                f"artifact={artifact} test_loss {init_loss:.4f}, test_top1 {init_top1:.4f}, test_top5 {init_top5:.4f}, "
                f"global_norm {sig['global_norm']:.4f}, classifier_norm {sig['classifier_norm']:.4f}",
                prefix="retrain",
            )
            self._append_per_client_metric(
                -1,
                idx,
                source=source,
                artifact_path=artifact,
                init_test_loss=float(init_loss),
                init_test_top1=float(init_top1),
                init_test_top5=float(init_top5),
                model_global_norm=float(sig["global_norm"]),
                model_global_mean=float(sig["global_mean"]),
                model_global_std=float(sig["global_std"]),
                classifier_norm=float(sig["classifier_norm"]),
            )
            client_round_prefix = f"task_{client.task_id}_client_{idx}_init"
            self.writerTf.add_scalar(client_round_prefix + "_test_loss", init_loss, 0)
            self.writerTf.add_scalar(client_round_prefix + "_test_top1", init_top1, 0)
            self.writerTf.add_scalar(client_round_prefix + "_test_top5", init_top5, 0)

    
    def run(self):
        print('self.config.resume: ', self.config.resume)
        baseline_method = str(getattr(self.config, "baseline_method", "fedavg")).lower().replace("-", "_")
        retrain_fedavg = bool(getattr(self.config, "retrain_fedavg", False))
        use_ditto = retrain_fedavg and baseline_method == "ditto"
        use_fedweit = retrain_fedavg and baseline_method == "fedweit"
        fedweit_personal_keys = set()
        if use_fedweit:
            raw_keys = getattr(self.config, "fedweit_personal_keys", "")
            if isinstance(raw_keys, str):
                fedweit_personal_keys = {k.strip() for k in raw_keys.split(",") if k.strip()}
            elif isinstance(raw_keys, (list, tuple, set)):
                fedweit_personal_keys = {str(k).strip() for k in raw_keys if str(k).strip()}
        ditto_mu = float(getattr(self.config, "ditto_mu", 0.01))
        if retrain_fedavg:
            self.write_log("[Retrain] FedAvg aggregation is enabled for this stage", prefix="retrain")
        if use_fedweit:
            self.write_log(
                f"[Retrain] baseline_method=fedweit, keep personalized keys: {sorted(fedweit_personal_keys)}",
                prefix="retrain",
            )
        if use_ditto:
            self.write_log(
                f"[Retrain] baseline_method=ditto, proximal_mu={ditto_mu:.6f}",
                prefix="retrain",
            )
        if self.config.resume:
            try:
                print('loading personalized client checkpoints:')
                loaded_rounds = []
                for id in self.clients_idx_arr:
                    self.clients[id].load_model()
                    self.clients[id].load_clients_opt()
                    loaded_rounds.append(getattr(self.clients[id], "round", 0))
                self.global_run_manager.round = max(loaded_rounds) if loaded_rounds else 0
            except Exception as e:
                print('Exception about load clients opt:', e)

        self.start_round = self.global_run_manager.round
        # 如果不是断点恢复，将 round 重置为 0，避免继承旧 checkpoint 的 round 导致轮数偏移
        if not getattr(self.config, "resume", False):
            self.global_run_manager.round = 0
            for idx in self.clients_idx_arr:
                self.clients[idx].round = 0
            self.start_round = 0
        print(f"[Retrain] start_round={self.start_round}, last_round={self.last_round}")
        self._log_retrain_init_snapshot()
        ditto_personal_states = None
        if use_ditto:
            ditto_personal_states = [
                copy.deepcopy(self.clients[idx].net.module.state_dict())
                for idx in self.clients_idx_arr
            ]
        for round in range(self.start_round, self.last_round):
            print('round', round+1)
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            round_wall_start = time.time()
            local_compute_time = 0.0
            aggregation_time = 0.0
            evaluation_time = 0.0
            checkpoint_io_time = 0.0
            round_time = time.time()
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            local_compute_start = time.time()
            server_model = copy.deepcopy(self.global_run_manager.net) if retrain_fedavg else None
            clients_params_arr = []
            clients_data_w = []
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[idx].train_run_manager(
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    server_model=server_model,
                    writer=self.writerTf,
                    global_round_idx=round + 1,
                    preserve_local_model=(
                        round == self.start_round
                        and bool(getattr(self.clients[idx], "preserve_local_model_for_first_sync", False))
                    ),
                )
                if round == self.start_round:
                    self.clients[idx].preserve_local_model_for_first_sync = False
                client_round_prefix = f"task_{self.clients[idx].task_id}_client_{idx}_round"
                self.writerTf.add_scalar(client_round_prefix + "_trn_loss", trn_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_trn_top1", trn_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_trn_top5", trn_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_val_loss", val_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_val_top1", val_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_val_top5", val_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_lr", lr, round)
                self.write_log(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    + " retrain client{} round{} trn_loss {:.4f}, trn_top1 {:.4f}, val_loss {:.4f}, val_top1 {:.4f}, lr {:.4f}".format(
                        idx,
                        round + 1,
                        trn_loss,
                        trn_top1,
                        val_loss,
                        val_top1,
                        lr,
                    ),
                    prefix="retrain",
                )
                self._append_per_client_metric(
                    round,
                    idx,
                    trn_loss=trn_loss,
                    trn_top1=trn_top1,
                    trn_top5=trn_top5,
                    val_loss=val_loss,
                    val_top1=val_top1,
                    val_top5=val_top5,
                    lr=lr,
                )
                client_checkpoint = {
                    "round": round,
                    "state_dict": self.clients[idx].net.module.state_dict(),
                    f"{idx}_weight_optimizer": self.clients[idx].optimizer.state_dict(),
                }
                client_checkpoint_io_start = time.time()
                self.clients[idx].save_model(client_checkpoint, is_best=False)
                checkpoint_io_time += time.time() - client_checkpoint_io_start
                if retrain_fedavg:
                    local_weight = self.clients[idx].get_local_data_weight()
                    if local_weight > 0:
                        clients_params_arr.append(copy.deepcopy(self.clients[idx].return_model_dict()))
                        clients_data_w.append(local_weight)
                    else:
                        self.write_log(
                            f"skip client {idx} in retrain round {round} aggregation because local data weight is 0",
                            prefix="retrain",
                        )
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
            local_compute_time += time.time() - local_compute_start
            if retrain_fedavg:
                aggregation_start = time.time()
                if len(clients_params_arr) == 0:
                    self.write_log(
                        f"retrain round {round} skip FedAvg because no client has positive sample weight",
                        prefix="retrain",
                    )
                else:
                    new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
                    if use_fedweit and fedweit_personal_keys:
                        for key in fedweit_personal_keys:
                            new_weight_fedavg.pop(key, None)
                    server_dict = self.global_run_manager.net.module.state_dict()
                    server_dict.update(new_weight_fedavg)
                    self.global_run_manager.net.module.load_state_dict(server_dict)
                    global_state = self.global_run_manager.net.module.state_dict()
                    for idx in self.clients_idx_arr:
                        if use_fedweit and fedweit_personal_keys:
                            local_state = self.clients[idx].net.module.state_dict()
                            merged_state = copy.deepcopy(global_state)
                            for key in fedweit_personal_keys:
                                if key in local_state:
                                    merged_state[key] = local_state[key]
                            self.clients[idx].net.module.load_state_dict(merged_state, strict=False)
                        else:
                            self.clients[idx].net.module.load_state_dict(global_state, strict=False)
                    global_checkpoint = {
                        "round": round,
                        "state_dict": self.global_run_manager.net.module.state_dict(),
                    }
                    global_ckpt_io_start = time.time()
                    self.global_run_manager.save_model(
                        global_checkpoint,
                        is_best=False,
                        model_name="global.pth.tar",
                    )
                    checkpoint_io_time += time.time() - global_ckpt_io_start
                aggregation_time += time.time() - aggregation_start
            if use_ditto and ditto_personal_states is not None:
                ditto_personal_start = time.time()
                global_reference_state = copy.deepcopy(self.global_run_manager.net.module.state_dict())
                for pos, idx in enumerate(self.clients_idx_arr):
                    self.clients[idx].net.module.load_state_dict(ditto_personal_states[pos], strict=False)
                    p_trn_loss, p_trn_top1, p_trn_top5, p_val_loss, p_val_top1, p_val_top5, p_lr = self.clients[idx].train_run_manager(
                        start_local_epoch=start_local_epoch,
                        last_local_epoch=last_local_epoch,
                        server_model=None,
                        writer=self.writerTf,
                        global_round_idx=round + 1,
                        preserve_local_model=True,
                        prox_reference_state=global_reference_state,
                        prox_mu=ditto_mu,
                    )
                    ditto_personal_states[pos] = copy.deepcopy(self.clients[idx].net.module.state_dict())
                    self.write_log(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        + " ditto_personal client{} round{} trn_loss {:.4f}, trn_top1 {:.4f}, val_loss {:.4f}, val_top1 {:.4f}, lr {:.4f}".format(
                            idx,
                            round + 1,
                            p_trn_loss,
                            p_trn_top1,
                            p_val_loss,
                            p_val_top1,
                            p_lr,
                        ),
                        prefix="retrain",
                    )
                    self._append_per_client_metric(
                        round,
                        idx,
                        ditto_personal=True,
                        trn_loss=float(p_trn_loss),
                        trn_top1=float(p_trn_top1),
                        trn_top5=float(p_trn_top5),
                        val_loss=float(p_val_loss),
                        val_top1=float(p_val_top1),
                        val_top5=float(p_val_top5),
                        lr=float(p_lr),
                    )
                    client_checkpoint = {
                        "round": round,
                        "state_dict": self.clients[idx].net.module.state_dict(),
                        f"{idx}_weight_optimizer": self.clients[idx].optimizer.state_dict(),
                    }
                    ditto_ckpt_start = time.time()
                    self.clients[idx].save_model(client_checkpoint, is_best=False)
                    checkpoint_io_time += time.time() - ditto_ckpt_start
                local_compute_time += time.time() - ditto_personal_start
            self.writerTf.add_scalar('clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('clients_local_test_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('clients_local_test_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('clients_local_test_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('clients_lr', clients_lr.avg, round)
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('round_time_use', round_time_use, round)
            # 日志写入加上时间戳
            self.write_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " +
                ' retrain clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_lr {:.4f}, round_time_use {:.4f}'.format(
                    clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg,
                    clients_lr.avg,
                    round_time_use),
                prefix='retrain')
            evaluation_start = time.time()
            avg_test_loss, avg_test_top1, avg_test_top5 = self.test_inference(
                tag="learned_net",
                round_idx=round,
            )
            self.writerTf.add_scalar('global_test_loss', avg_test_loss, round)
            self.writerTf.add_scalar('global_test_top1', avg_test_top1, round)
            self.writerTf.add_scalar('global_test_top5', avg_test_top5, round)
            evaluation_time += time.time() - evaluation_start
            checkpoint_io_time += self._emit_progress(
                "retrain_round_completed",
                round_idx=round,
                checkpoint_path=os.path.join(self.path, "clients"),
            )
            self.global_run_manager.round = round + 1
            wall_clock_time = time.time() - round_wall_start
            algorithm_time = local_compute_time + aggregation_time + evaluation_time
            runtime_context = getattr(self.config, "runtime_context", None)
            if runtime_context is not None:
                runtime_context.update_timing(
                    "retrain",
                    round_idx=round,
                    local_compute_time=local_compute_time,
                    aggregation_time=aggregation_time,
                    evaluation_time=evaluation_time,
                    checkpoint_io_time=checkpoint_io_time,
                    algorithm_time=algorithm_time,
                    wall_clock_time=wall_clock_time,
                )
            self.writerTf.add_scalar('retrain_local_compute_time_min', local_compute_time / 60, round)
            self.writerTf.add_scalar('retrain_aggregation_time_min', aggregation_time / 60, round)
            self.writerTf.add_scalar('retrain_evaluation_time_min', evaluation_time / 60, round)
            self.writerTf.add_scalar('retrain_checkpoint_io_time_min', checkpoint_io_time / 60, round)
            self.writerTf.add_scalar('retrain_algorithm_time_min', algorithm_time / 60, round)
            self.writerTf.add_scalar('retrain_wall_clock_time_min', wall_clock_time / 60, round)
            self.write_log(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " +
                ' retrain_timing local_compute {:.4f}m, aggregation {:.4f}m, evaluation {:.4f}m, checkpoint_io {:.4f}m, algorithm {:.4f}m, wall {:.4f}m'.format(
                    local_compute_time / 60,
                    aggregation_time / 60,
                    evaluation_time / 60,
                    checkpoint_io_time / 60,
                    algorithm_time / 60,
                    wall_clock_time / 60,
                ),
                prefix='retrain')
        self.writerTf.close()

    
    def get_server(self):
        return copy.deepcopy(self.global_run_manager)

    
    def get_model(self):
        return copy.deepcopy(self.global_run_manager.net.module)

    
    def get_weights(self):
        return copy.deepcopy(self.global_run_manager.net.module.state_dict())

    
    def test_inference(self, tag=None, round_idx=None):
        # Test inference after completion of training. For personalized retrain, report cluster-average client accuracy.
        client_losses = AverageMeter()
        client_top1 = AverageMeter()
        client_top5 = AverageMeter()
        per_client_top1 = []
        per_client_loss = []
        per_client_top5 = []
        per_client_test_weight = []
        for idx in self.clients_idx_arr:
            val_loss, val_acc_top1, val_acc_top5 = self.clients[idx].validate(is_test=True, return_top5=True)
            client_losses.update(val_loss)
            client_top1.update(val_acc_top1)
            client_top5.update(val_acc_top5)
            per_client_loss.append(float(val_loss))
            per_client_top1.append(float(val_acc_top1))
            per_client_top5.append(float(val_acc_top5))
            test_weight = int(getattr(self.clients[idx].run_config.data_provider, "tst_set_length", 0))
            per_client_test_weight.append(test_weight)
            self._append_per_client_metric(
                round_idx if round_idx is not None else -1,
                idx,
                test_loss=float(val_loss),
                test_top1=float(val_acc_top1),
                test_top5=float(val_acc_top5),
                test_weight=test_weight,
            )
            try:
                self.clients[idx].write_log(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] client{idx},test_eval loss {val_loss:.4f}, top1 {val_acc_top1:.4f}, top5 {val_acc_top5:.4f}",
                    prefix='test',
                    should_print=False,
                )
            except Exception:
                pass
        prefix_name = tag if tag is not None else self.hardware
        self.write_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " +
            "{},test_eval loss {:.4f}, top1 {:.4f}, top5 {:.4f}".format(
                prefix_name, client_losses.avg, client_top1.avg, client_top5.avg 
            ),
            prefix='test',
            should_print=True
        )
        if per_client_top1:
            arr_top1 = np.array(per_client_top1, dtype=np.float64)
            arr_loss = np.array(per_client_loss, dtype=np.float64)
            arr_top5 = np.array(per_client_top5, dtype=np.float64)
            arr_weight = np.array(per_client_test_weight, dtype=np.float64)
            weighted_top1 = float(np.average(arr_top1, weights=arr_weight)) if np.sum(arr_weight) > 0 else float(np.mean(arr_top1))
            stats_record = {
                "client_count": int(len(arr_top1)),
                "test_top1_mean": float(np.mean(arr_top1)),
                "test_top1_std": float(np.std(arr_top1)),
                "test_top1_min": float(np.min(arr_top1)),
                "test_top1_max": float(np.max(arr_top1)),
                "test_top1_weighted_mean": weighted_top1,
                "test_loss_mean": float(np.mean(arr_loss)),
                "test_top5_mean": float(np.mean(arr_top5)),
            }
            self._append_test_top1_stats(
                round_idx if round_idx is not None else -1,
                prefix_name,
                stats_record,
            )
            self.write_log(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                + "clients_test_top1_stats mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}, weighted_mean {:.4f}".format(
                    stats_record["test_top1_mean"],
                    stats_record["test_top1_std"],
                    stats_record["test_top1_min"],
                    stats_record["test_top1_max"],
                    stats_record["test_top1_weighted_mean"],
                ),
                prefix='test',
                should_print=True,
            )
        return client_losses.avg, client_top1.avg, client_top5.avg

    
    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    @property

    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path


def arrange_local_epoch_from_round(global_round=0, local_epoch_number=10):
    return global_round * local_epoch_number, (global_round + 1) * local_epoch_number
