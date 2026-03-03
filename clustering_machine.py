import pickle
import time
import warnings
import json

import numpy as np
import torch
from arch_prior import apply_fused_arch_parameters, append_arch_prior_log
from utils_old import save_checkpoint

from nas_manager import ArchSearchRunManager
import time
from utils_old import *
# from models import *
from run_manager import *
from models.super_nets.super_proxyless import *
from tensorboardX import SummaryWriter


class ClusteringMachine:
    def __init__(self, target_hardware=None, config=None, global_server=None,
                 clients_idx_arr=None, clients=None, start_round=0,
                 last_round=None, path='./', task_id=1):
        self.hardware = target_hardware
        self.config = config
        self.global_server = copy.deepcopy(global_server)
        self.clients_idx_arr = clients_idx_arr
        self.clients = clients
        self.start_round = start_round
        self.last_round = last_round
        self.local_epoch_number = config.local_epoch_number
        self.path = path
        self.task_id = task_id
        self._logs_path, self._save_path = None, None
        
        # TensorBoard 日志路径中加入 task_id
        if self.hardware is not None:
            self.writerTf = SummaryWriter(logdir=os.path.join(self.path, 'tensorboard'), 
                                          comment=f"{self.hardware}_fed_search_task_{task_id}")
        else:
            self.writerTf = SummaryWriter(logdir=os.path.join(self.path, 'tensorboard'), 
                                          comment=f"fed_search_task_{task_id}")
        print('tensorboardX logdir', self.writerTf.logdir)

    def _append_per_client_metric(self, phase, round_idx, client_id, **metrics):
        record = {
            "phase": phase,
            "task_id": int(self.task_id),
            "round": int(round_idx),
            "client_id": int(client_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        record.update(metrics)
        metrics_path = os.path.join(self.logs_path, "per_client_metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as fout:
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
                print(f"[Progress] search callback failed for {event}: {e}")
            return max(0.0, time.time() - start)
        return 0.0

    
    def train_clients(self):
        self.start_round = self.global_server.round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        best_val_acc = 0
        self._emit_progress("search_started")
        for round in range(self.start_round, self.last_round):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_entropy, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            round_wall_start = time.time()
            local_compute_time = 0.0
            aggregation_time = 0.0
            evaluation_time = 0.0
            checkpoint_io_time = 0.0
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            # 拿当前全局超网权重，作为本轮下发给各客户端的初始模型
            server_model = copy.deepcopy(self.global_server.net)
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            local_compute_start = time.time()
            for idx in self.clients_idx_arr:
                if round == self.start_round:
                    prior_state = getattr(self.clients[idx], "arch_prior_state", None)
                    if prior_state is not None and not bool(prior_state.get("applied", False)):
                        apply_stats = apply_fused_arch_parameters(
                            self.clients[idx].net,
                            prior_state.get("fused_arch_params"),
                        )
                        prior_state["applied"] = apply_stats is not None
                        prior_state["apply_stats"] = apply_stats
                        self.clients[idx].preserve_local_arch_params_for_first_sync = bool(apply_stats is not None)
                        if bool(getattr(self.config, "log_arch_prior_details", True)):
                            append_arch_prior_log(
                                os.path.join(self.logs_path, "arch_prior_details.jsonl"),
                                {
                                    "event": "prior_applied",
                                    "task_id": int(self.task_id),
                                    "client_id": int(idx),
                                    "round": int(round),
                                    "selected_task_ids": prior_state.get("selected_task_ids", []),
                                    "weights": prior_state.get("weights").tolist()
                                    if prior_state.get("weights") is not None
                                    else None,
                                    **(apply_stats or {}),
                                },
                            )
                        if apply_stats is not None:
                            self.write_log(
                                "arch_prior client{} round{} selected={} alpha_diff_norm {:.6f}".format(
                                    idx,
                                    round + 1,
                                    prior_state.get("selected_task_ids", []),
                                    apply_stats.get("alpha_diff_norm", 0.0),
                                ),
                                prefix="search",
                            )
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, entropy, lr = self.clients[
                    idx].train(
                    server_model=server_model,
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    writer=self.writerTf,
                    preserve_local_arch_params=(
                        round == self.start_round
                        and bool(getattr(self.clients[idx], "preserve_local_arch_params_for_first_sync", False))
                    ),
                )
                if round == self.start_round:
                    self.clients[idx].preserve_local_arch_params_for_first_sync = False
                # local_weight 表示该客户端本轮可用的训练样本数，聚合时作为加权系数
                local_weight = self.clients[idx].get_local_data_weight()
                if local_weight <= 0:
                    # 跳过无数据客户端，避免聚合时 total_weight=0 -> NaN
                    self.write_log(f"skip client {idx} in round {round} because local data weight is 0", prefix='search')
                    continue
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(local_weight)
                client_round_prefix = f"task_{self.task_id}_client_{idx}_round"
                self.writerTf.add_scalar(client_round_prefix + "_search_trn_loss", trn_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_trn_top1", trn_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_trn_top5", trn_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_val_loss", val_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_val_top1", val_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_val_top5", val_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_entropy", entropy, round)
                self.writerTf.add_scalar(client_round_prefix + "_search_lr", lr, round)
                self.write_log(
                    "[{}] search client{} round{} trn_loss {:.4f}, trn_top1 {:.4f}, val_loss {:.4f}, val_top1 {:.4f}, entropy {:.4f}, lr {:.4f}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        idx,
                        round + 1,
                        trn_loss,
                        trn_top1,
                        val_loss,
                        val_top1,
                        entropy,
                        lr,
                    ),
                    prefix='search',
                )
                self._append_per_client_metric(
                    "search",
                    round,
                    idx,
                    trn_loss=trn_loss,
                    trn_top1=trn_top1,
                    trn_top5=trn_top5,
                    val_loss=val_loss,
                    val_top1=val_top1,
                    val_top5=val_top5,
                    entropy=entropy,
                    lr=lr,
                )
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_entropy.update(entropy)
                clients_lr.update(lr)
            local_compute_time += time.time() - local_compute_start
            self.writerTf.add_scalar('clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('clients_val_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('clients_val_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('clients_val_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('clients_entropy', clients_entropy.avg, round)
            self.writerTf.add_scalar('clients_lr', clients_lr.avg, round)
            self.write_log(
                'search clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_entropy {:.4f}, clients_lr {:.4f}'.format(
                    clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg, clients_entropy.avg,
                    clients_lr.avg),
                prefix='search')
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('round_time_use', round_time_use, round)

            # update new_fedavg_weight
            if len(clients_params_arr) == 0:
                self.write_log(f"train round {round} skip aggregation because no client has data", prefix='search')
                continue
            
            # 聚合后直接落到 DataParallel 内部的 module，确保键名一致且不丢失权重
            aggregation_start = time.time()
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.run_manager.net.module.load_state_dict(server_dict)
            aggregation_time += time.time() - aggregation_start
            
            self.global_server.write_log('-' * 30 + 'Current Architecture [%d]' % (round + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.global_server.net.blocks):
                self.global_server.write_log('%d. %s' % (idx, block.module_str), prefix='arch')

            # Calculate avg training accuracy over all users at every round
            evaluation_start = time.time()
            self.global_server.net.eval()
            with torch.no_grad():
                # directly use global_server's centralized data for faster inference
                val_loss, val_acc_top1, val_acc_top5 = self.global_server.inference(is_test=False)
                if best_val_acc < val_acc_top1:
                    best_val_acc = val_acc_top1
                    is_best = True
                else:
                    is_best = False
            evaluation_time += time.time() - evaluation_start
            # save global model and each client's opt.
            checkpoint = {}
            checkpoint['round'] = round
            checkpoint['warmup'] = False
            checkpoint['state_dict'] = self.global_server.net.state_dict()
            # 保存全局 server 的优化器状态，便于跨任务继承
            checkpoint["server_weight_optimizer"] = self.global_server.run_manager.optimizer.state_dict()
            checkpoint["server_arch_optimizer"] = self.global_server.arch_optimizer.state_dict()
            for id in self.clients_idx_arr:
                checkpoint[f"task_{self.task_id}_{id}_weight_optimizer"] = self.clients[id].run_manager.optimizer.state_dict()
                checkpoint[f"task_{self.task_id}_{id}_arch_optimizer"] = self.clients[id].arch_optimizer.state_dict()
            checkpoint_io_start = time.time()
            self.global_server.run_manager.save_model(checkpoint, is_best=is_best, model_name="global.pth.tar")
            checkpoint_io_time += time.time() - checkpoint_io_start
            checkpoint_io_time += self._emit_progress(
                "search_round_completed",
                round_idx=round,
                checkpoint_path=os.path.join(self.global_server.run_manager.save_path, "global.pth.tar"),
            )
            wall_clock_time = time.time() - round_wall_start
            algorithm_time = local_compute_time + aggregation_time + evaluation_time
            runtime_context = getattr(self.config, "runtime_context", None)
            if runtime_context is not None:
                runtime_context.update_timing(
                    "search",
                    round_idx=round,
                    local_compute_time=local_compute_time,
                    aggregation_time=aggregation_time,
                    evaluation_time=evaluation_time,
                    checkpoint_io_time=checkpoint_io_time,
                    algorithm_time=algorithm_time,
                    wall_clock_time=wall_clock_time,
                )
            self.writerTf.add_scalar('search_local_compute_time_min', local_compute_time / 60, round)
            self.writerTf.add_scalar('search_aggregation_time_min', aggregation_time / 60, round)
            self.writerTf.add_scalar('search_evaluation_time_min', evaluation_time / 60, round)
            self.writerTf.add_scalar('search_checkpoint_io_time_min', checkpoint_io_time / 60, round)
            self.writerTf.add_scalar('search_algorithm_time_min', algorithm_time / 60, round)
            self.writerTf.add_scalar('search_wall_clock_time_min', wall_clock_time / 60, round)
            self.write_log(
                'search_timing local_compute {:.4f}m, aggregation {:.4f}m, evaluation {:.4f}m, checkpoint_io {:.4f}m, algorithm {:.4f}m, wall {:.4f}m'.format(
                    local_compute_time / 60,
                    aggregation_time / 60,
                    evaluation_time / 60,
                    checkpoint_io_time / 60,
                    algorithm_time / 60,
                    wall_clock_time / 60,
                ),
                prefix='search',
            )
                
            # self.test_inference()  # 测试集上跑一下
        self._emit_progress(
            "search_completed",
            checkpoint_path=os.path.join(self.global_server.run_manager.save_path, "global.pth.tar"),
        )
        self.writerTf.close()

    
    def warmup_clients(self):
        self.warmup_round = self.global_server.warmup_round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        self._emit_progress("warmup_started")
        for round in range(self.warmup_round, self.config.warmup_n_rounds):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            round_wall_start = time.time()
            local_compute_time = 0.0
            aggregation_time = 0.0
            checkpoint_io_time = 0.0
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            server_model = copy.deepcopy(self.global_server.net)
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            local_compute_start = time.time()
            for idx in self.clients_idx_arr:
                # 预热阶段客户端的训练：随机子网前后向，只更新权重，不更新架构参数
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[
                    idx].warm_up(server_model=server_model, start_local_epoch=start_local_epoch,
                                 last_local_epoch=last_local_epoch, writer=self.writerTf)
                local_weight = self.clients[idx].get_local_data_weight()
                if local_weight <= 0:
                    self.write_log(f"skip client {idx} in warmup round {round} because local data weight is 0", prefix='warmup')
                    continue
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(local_weight)
                client_round_prefix = f"task_{self.task_id}_client_{idx}_round"
                self.writerTf.add_scalar(client_round_prefix + "_warmup_trn_loss", trn_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_trn_top1", trn_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_trn_top5", trn_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_val_loss", val_loss, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_val_top1", val_top1, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_val_top5", val_top5, round)
                self.writerTf.add_scalar(client_round_prefix + "_warmup_lr", lr, round)
                self.write_log(
                    "[{}] warmup client{} round{} trn_loss {:.4f}, trn_top1 {:.4f}, val_loss {:.4f}, val_top1 {:.4f}, lr {:.4f}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        idx,
                        round + 1,
                        trn_loss,
                        trn_top1,
                        val_loss,
                        val_top1,
                        lr,
                    ),
                    prefix='warmup',
                )
                self._append_per_client_metric(
                    "warmup",
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
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
            local_compute_time += time.time() - local_compute_start
            self.writerTf.add_scalar('warmup clients_trn_loss', clients_trn_loss.avg, round)
            self.writerTf.add_scalar('warmup clients_trn_top1', clients_trn_top1.avg, round)
            self.writerTf.add_scalar('warmup clients_trn_top5', clients_trn_top5.avg, round)
            self.writerTf.add_scalar('warmup clients_val_loss', clients_val_loss.avg, round)
            self.writerTf.add_scalar('warmup clients_val_top1', clients_val_top1.avg, round)
            self.writerTf.add_scalar('warmup clients_val_top5', clients_val_top5.avg, round)
            self.writerTf.add_scalar('warmup clients_lr', clients_lr.avg, round)
            self.write_log('warmup clients_trn_loss {:.4f}, clients_trn_top1 {:.4f}, clients_val_loss {:.4f}, clients_val_top1 {:.4f}, clients_lr {:.4f}'.format(clients_trn_loss.avg, clients_trn_top1.avg, clients_val_loss.avg, clients_val_top1.avg,clients_lr.avg),prefix='warmup')
            round_time_use = (time.time() - round_time) / 60
            self.writerTf.add_scalar('warmup round_time_use', round_time_use, round)
            if len(clients_params_arr) == 0:
                self.write_log(f"warmup round {round} skip aggregation because no client has data", prefix='warmup')
                continue
            
            # 同样使用 DataParallel 内部 module 的 state_dict 保持键名一致
            aggregation_start = time.time()
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.run_manager.net.module.load_state_dict(server_dict)
            aggregation_time += time.time() - aggregation_start
            
            # 保存聚合后的超网权重 + 各优化器状态，便于断点续训/跨任务继承
            checkpoint = {}
            checkpoint['warmup_round'] = round
            checkpoint['warmup'] = True
            checkpoint['state_dict'] = self.global_server.net.state_dict()
            checkpoint["server_weight_optimizer"] = self.global_server.run_manager.optimizer.state_dict()
            checkpoint["server_arch_optimizer"] = self.global_server.arch_optimizer.state_dict()
            for id in self.clients_idx_arr:
                checkpoint[f"task_{self.task_id}_{id}_weight_optimizer"] = self.clients[id].run_manager.optimizer.state_dict()
                checkpoint[f"task_{self.task_id}_{id}_arch_optimizer"] = self.clients[id].arch_optimizer.state_dict()
            checkpoint_io_start = time.time()
            self.global_server.run_manager.save_model(checkpoint, model_name="warmup.pth.tar")
            checkpoint_io_time += time.time() - checkpoint_io_start
            checkpoint_io_time += self._emit_progress(
                "warmup_round_completed",
                round_idx=round,
                checkpoint_path=os.path.join(self.global_server.run_manager.save_path, "warmup.pth.tar"),
            )
            wall_clock_time = time.time() - round_wall_start
            algorithm_time = local_compute_time + aggregation_time
            runtime_context = getattr(self.config, "runtime_context", None)
            if runtime_context is not None:
                runtime_context.update_timing(
                    "warmup",
                    round_idx=round,
                    local_compute_time=local_compute_time,
                    aggregation_time=aggregation_time,
                    evaluation_time=0.0,
                    checkpoint_io_time=checkpoint_io_time,
                    algorithm_time=algorithm_time,
                    wall_clock_time=wall_clock_time,
                )
            self.writerTf.add_scalar('warmup_local_compute_time_min', local_compute_time / 60, round)
            self.writerTf.add_scalar('warmup_aggregation_time_min', aggregation_time / 60, round)
            self.writerTf.add_scalar('warmup_checkpoint_io_time_min', checkpoint_io_time / 60, round)
            self.writerTf.add_scalar('warmup_algorithm_time_min', algorithm_time / 60, round)
            self.writerTf.add_scalar('warmup_wall_clock_time_min', wall_clock_time / 60, round)
            self.write_log(
                'warmup_timing local_compute {:.4f}m, aggregation {:.4f}m, checkpoint_io {:.4f}m, algorithm {:.4f}m, wall {:.4f}m'.format(
                    local_compute_time / 60,
                    aggregation_time / 60,
                    checkpoint_io_time / 60,
                    algorithm_time / 60,
                    wall_clock_time / 60,
                ),
                prefix='warmup',
            )
            
        checkpoint = {}
        checkpoint['warmup_round'] = self.config.warmup_n_rounds
        checkpoint['warmup'] = False
        checkpoint['state_dict'] = self.global_server.net.state_dict()
        for id in self.clients_idx_arr:
            checkpoint[f"task_{self.task_id}_{id}_weight_optimizer"] = self.clients[id].run_manager.optimizer.state_dict()
            checkpoint[f"task_{self.task_id}_{id}_arch_optimizer"] = self.clients[id].arch_optimizer.state_dict()
        checkpoint_io_start = time.time()
        self.global_server.run_manager.save_model(checkpoint, model_name=f"warmup.pth.tar")
        checkpoint_io_time = time.time() - checkpoint_io_start
        checkpoint_io_time += self._emit_progress(
            "warmup_completed",
            checkpoint_path=os.path.join(self.global_server.run_manager.save_path, "warmup.pth.tar"),
        )
        self.write_log(
            'warmup_finalize_checkpoint_io {:.4f}m'.format(checkpoint_io_time / 60),
            prefix='warmup',
        )
        self.writerTf.close()

    
    def run(self):
        if self.config.resume:
            try:
                # 先加载服务端模型权重与轮次
                self.global_server.load_model()
                # 读取 latest 指向的 ckpt，补充加载全局/客户端的优化器状态（带 task_id 前缀）
                latest_fname = os.path.join(self.global_server.run_manager.save_path, "latest.txt")
                model_fname = None
                if os.path.exists(latest_fname):
                    with open(latest_fname, "r") as fin:
                        model_fname = fin.readline().strip()
                if model_fname and os.path.exists(model_fname):
                    ckpt = torch.load(model_fname, map_location=torch.device("cpu"))
                    # 全局优化器
                    if "server_weight_optimizer" in ckpt:
                        try:
                            self.global_server.run_manager.optimizer.load_state_dict(ckpt["server_weight_optimizer"])
                        except Exception as e:
                            print(f"Failed to load server_weight_optimizer: {e}")
                    if "server_arch_optimizer" in ckpt and hasattr(self.global_server, "arch_optimizer"):
                        try:
                            self.global_server.arch_optimizer.load_state_dict(ckpt["server_arch_optimizer"])
                        except Exception as e:
                            print(f"Failed to load server_arch_optimizer: {e}")
                    # 各客户端优化器（注意键名带 task_id 前缀）
                    for cid in self.clients_idx_arr:
                        w_key = f"task_{self.task_id}_{cid}_weight_optimizer"
                        a_key = f"task_{self.task_id}_{cid}_arch_optimizer"
                        if w_key in ckpt:
                            try:
                                self.clients[cid].run_manager.optimizer.load_state_dict(ckpt[w_key])
                            except Exception as e:
                                print(f"Failed to load client {cid} weight_optimizer: {e}")
                        if a_key in ckpt and hasattr(self.clients[cid], "arch_optimizer"):
                            try:
                                self.clients[cid].arch_optimizer.load_state_dict(ckpt[a_key])
                            except Exception as e:
                                print(f"Failed to load client {cid} arch_optimizer: {e}")
                else:
                    print("No latest checkpoint found for resume; only weights and round were restored.")
            except Exception as e:
                print('Exception about load clients opt:', e)
                
        # 先warmup，再train
        skip_warmup = bool(getattr(self.config, "skip_warmup", False))
        if getattr(self.config, "resume", False) and not getattr(self.global_server, "warmup", False):
            skip_warmup = True
        if not skip_warmup:
            self.warmup_clients()
        self.train_clients()

    
    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        # 添加当前时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_str = f"[{timestamp}] {log_str}"
        
        log_file = os.path.join(self.logs_path, f'task_{self.task_id}_{prefix}.log')  # 日志文件名包含 task_id
        with open(log_file, 'a') as fout:
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
    def get_server(self):
        return copy.deepcopy(self.global_server)

    
    def transfer_to_normal_net(self):
        server_model = copy.deepcopy(self.global_server.net)
        if isinstance(server_model, nn.DataParallel):
            server_model = server_model.module

        normal_net = server_model.cpu().convert_to_normal_net()
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
        os.makedirs(os.path.join(self.global_server.run_manager.path, self.hardware + '_learned_net'), exist_ok=True)
        torch.save(
            {'state_dict': normal_net.state_dict()},
            os.path.join(self.global_server.run_manager.path, self.hardware + '_learned_net/init.pth.tar')
        )

    
    def test_inference(self):
        # Test inference after completion of training
        AvgTestLoss, TestAccuracyTop1, TestAccuracyTop5 = self.global_server.inference(server_model=self.global_server.net,
                                                                                      is_test=True,
                                                                                      return_top5=True)
        print("|---- Avg Test Loss: {:.2f}%".format(AvgTestLoss))
        print("|---- Test Accuracy Top1: {:.2f}%".format(TestAccuracyTop1))
        print('{}, Test Accuracy Top1 :{:.4f}, Test Loss:{:.4f}'.format(self.hardware,
                                                                        TestAccuracyTop1 * 0.01,
                                                                        AvgTestLoss * 0.01))
        

        # 单独记录 test_eval，方便对齐其他指标
        self.write_log(
            "{},test_eval loss {:.4f}, top1 {:.4f}, top5 {:.4f}".format(
                self.hardware, AvgTestLoss, TestAccuracyTop1, TestAccuracyTop5
            ),
            prefix='test',
        )
        self.writerTf.add_scalar('TestAccuracyTop1', TestAccuracyTop1)
        self.writerTf.add_scalar('ValLAvgTestLoss', TestAccuracyTop1)



def arrange_local_epoch_from_round(global_round=0, local_epoch_number=10):
    return global_round * local_epoch_number, (global_round + 1) * local_epoch_number
