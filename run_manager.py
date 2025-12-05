# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import torch.nn.parallel
import torch.optim

import time
import json
from datetime import timedelta
import numpy as np
import copy

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from utils import *
from models.normal_nets.proxyless_nets import ProxylessNASNets
from modules.mix_op import MixedEdge
import torch.nn.functional as F

# run_manager is used by nas_manager

"""
    RunConfig:运行配置基类
"""
class RunConfig(object):

    def __init__(
        self,
        client_id,
        dataset_location,
        n_epochs,
        init_lr,
        lr_schedule_type,
        lr_schedule_param,
        dataset,
        train_batch_size,
        test_batch_size,
        valid_size,
        opt_type,
        opt_param,
        weight_decay,
        label_smoothing,
        no_decay_keys,
        model_init,
        init_div_groups,
        validation_frequency,
        print_frequency,
        search,
        task_id,
        is_client
    ):
        self.client_id = client_id
        self.task_id = task_id
        self.dataset_location = dataset_location
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency
        self.search = search
        self.is_client = is_client

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate相关 """
    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == "cosine":
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))

            # 前25个epoch线性增长
            if epoch < 25:
                lr = ((epoch + 1) / 25) * self.init_lr
        else:
            raise ValueError("do not support: %s" % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """adjust learning of a given optimizer and return the new learning rate"""
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    """ data provider相关 """
    @property
    def data_config(self):
        raise NotImplementedError

    # 动态加载数据
    @property
    def data_provider(self):
        if self._data_provider is None:
            # 根据 dataset 决定使用 CIFAR-10 或 CIFAR-100 的 FCL 数据划分
            if str(self.dataset).lower() in ["cifar10", "cifar"]:
                from data_providers.cifar10_fcl_dirichlet_split import CifarDataProvider10
                self._data_provider = CifarDataProvider10(**self.data_config)
            else:
                from data_providers.cifar100_fcl_dirichlet_split import CifarDataProvider100
                self._data_provider = CifarDataProvider100(**self.data_config)

        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train()

    @property
    def valid_loader(self):
        return self.data_provider.valid()

    @property
    def test_loader(self):
        return self.data_provider.test()

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data

    """ optimizer相关 """
    def build_optimizer(self, net_params):
        if self.opt_type == "sgd":
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get("momentum", 0.9), opt_param.get(
                "nesterov", True
            )
            if self.no_decay_keys:
                optimizer = torch.optim.SGD(
                    [
                        {"params": net_params[0], "weight_decay": self.weight_decay},
                        {"params": net_params[1], "weight_decay": 0},
                    ],
                    lr=self.init_lr,
                    momentum=momentum,
                    nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.SGD(
                    net_params,
                    self.init_lr,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=self.weight_decay,
                )
        else:
            raise NotImplementedError
        return optimizer


"""
    RunManager:运行管理基类
"""


class RunManager:

    def __init__(
        self,
        path,
        net,
        run_config: RunConfig,
        out_log=True,
        task_id=1,
        measure_latency=None,
        init_model=True,
    ):
        print("RunManager初始化开始...")
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log
        self.task_id = task_id

        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0

        # 部分场景需要复用外部加载的权重，此时可跳过重新初始化
        if init_model:
            self.net.init_model(run_config.model_init, run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency model
        self.net_on_cpu_for_latency = copy.deepcopy(self.net).cpu()
        self.latency_estimator = LatencyEstimator()
        self.start_epoch = 0
        self.round = 0
        # 最近一次测试的分任务精度，用于上层日志/遗忘分析
        self.last_task_acc = None
        # 用于 EWC 正则的快照与 Fisher 信息
        self.ewc_prev_params = None
        self.ewc_fisher = None
        self.ewc_lambda = float(getattr(run_config, "ewc_lambda", 0.0))
        self.ewc_samples_per_task = int(getattr(run_config, "ewc_samples_per_task", 0))
        self.ewc_online_interval = int(getattr(run_config, "ewc_online_interval", 0))
        self.cl_reg_method = str(getattr(run_config, "cl_reg_method", "mas")).lower()
        self.rwalk_path_score = None
        self.cl_reg_decay = float(getattr(run_config, "cl_reg_decay", 1.0))
        self.cl_reg_clip = getattr(run_config, "cl_reg_clip", None)
        self.cl_penalty_clip = getattr(run_config, "cl_penalty_clip", None)
        # move network to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        else:
            print("can not use GPU!")
            raise ValueError
        # net info
        self.print_net_info(measure_latency)

        self.criterion = nn.CrossEntropyLoss()
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split("#")
            self.optimizer = self.run_config.build_optimizer(
                [
                    self.net.module.get_parameters(
                        keys, mode="exclude"
                    ),  # parameters with weight decay
                    self.net.module.get_parameters(
                        keys, mode="include"
                    ),  # parameters without weight decay
                ]
            )
        else:
            self.optimizer = self.run_config.build_optimizer(
                self.net.module.weight_parameters()
            )    
        
        # KD 超参
        self.cl_kd_method = str(getattr(run_config, "cl_kd_method", "none")).lower()
        self.cl_kd_logit_lambda = float(getattr(run_config, "cl_kd_logit_lambda", 0.0))
        self.cl_kd_temperature = float(getattr(run_config, "cl_kd_temperature", 2.0))
        self.cl_kd_conf_threshold = float(getattr(run_config, "cl_kd_conf_threshold", 0.5))
        self.teacher_model = None
        
        # 正交更新超参与状态
        self.cl_ortho_method = str(getattr(run_config, "cl_ortho_method", "none")).lower()
        self.cl_ortho_scale = float(getattr(run_config, "cl_ortho_scale", 1.0))
        self.ortho_samples_per_task = int(getattr(run_config, "ortho_samples_per_task", 0))
        self.ortho_kd_history_max = int(getattr(run_config, "ortho_kd_history_max", 10))

        # 存储旧任务的参考梯度方向（按参数名）
        self.ortho_ref_grads = None  # dict[name] = tensor on CPU
        # kd_ortho 专用：任务开始时预估的 KD 平均梯度
        self.kd_ortho_ref_grads = None  # dict[name] = tensor on CPU
        # prev_grad_ortho：上一轮梯度缓存（按参数名）
        self.prev_grads = None  # dict[name] = tensor on CPU
        # kd_prev_grad_ortho：上一轮梯度缓存（按参数名，需 teacher 支持）
        self.prev_kd_grads = None  # dict[name] = tensor on CPU
        self.prev_kd_grads_history = []  # list[dict], 维护多轮 KD 参考

    def load_ewc_state(self, state):
        """加载外部保存的 EWC 状态；state=None 时重置。"""
        if not state:
            self.ewc_prev_params, self.ewc_fisher = None, None
            self.rwalk_path_score = None
            return
        params = state.get("params")
        fisher = state.get("fisher")
        rwalk = state.get("rwalk_path_score")
        if params is None or fisher is None:
            self.ewc_prev_params, self.ewc_fisher = None, None
            self.rwalk_path_score = None
            return
        cur_params = dict(self.net.module.named_parameters())
        filtered_params, filtered_fisher, filtered_rwalk = {}, {}, {}
        dropped_params = dropped_fisher = 0

        for name, v in params.items():
            cur = cur_params.get(name)
            if cur is None or cur.shape != v.shape:
                dropped_params += 1
                continue
            filtered_params[name] = v.clone()

        for name, v in fisher.items():
            cur = cur_params.get(name)
            if cur is None or cur.shape != v.shape:
                dropped_fisher += 1
                continue
            filtered_fisher[name] = v.clone()

        if rwalk is not None:
            for name, v in rwalk.items():
                cur = cur_params.get(name)
                if cur is None or cur.shape != v.shape:
                    continue
                filtered_rwalk[name] = v.clone()

        if not filtered_params or not filtered_fisher:
            self.ewc_prev_params, self.ewc_fisher = None, None
            self.rwalk_path_score = None
            if dropped_params or dropped_fisher:
                print(f"[EWC] skip load due to shape mismatch, dropped params={dropped_params}, fishers={dropped_fisher}")
            return

        self.ewc_prev_params = filtered_params
        self.ewc_fisher = filtered_fisher
        self.rwalk_path_score = filtered_rwalk if filtered_rwalk else None
        if dropped_params or dropped_fisher:
            print(f"[EWC] loaded with shape filtering: params={len(filtered_params)}, fisher={len(filtered_fisher)}, dropped params={dropped_params}, fishers={dropped_fisher}")

    def export_ewc_state(self):
        if self.ewc_prev_params is None or self.ewc_fisher is None:
            return None
        return {
            "params": {k: v.clone() for k, v in self.ewc_prev_params.items()},
            "fisher": {k: v.clone() for k, v in self.ewc_fisher.items()},
            "rwalk_path_score": {k: v.clone() for k, v in self.rwalk_path_score.items()} if self.rwalk_path_score is not None else None,
        }

    def _ewc_penalty(self):
        """基于当前参数与历史快照计算 EWC 正则项。"""
        if (
            self.ewc_lambda <= 0
            or self.ewc_prev_params is None
            or self.ewc_fisher is None
        ):
            return None
        penalty = None
        for name, param in self.net.module.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.ewc_prev_params or name not in self.ewc_fisher:
                continue
            fisher = self.ewc_fisher[name].to(param.device)
            if self.cl_reg_method == "rwalk" and self.rwalk_path_score is not None:
                # RWalk 将路径积分的重要性与 Fisher 相加
                path_score = self.rwalk_path_score.get(name, None)
                if path_score is not None:
                    fisher = fisher + path_score.to(param.device)
            prev = self.ewc_prev_params[name].to(param.device)
            term = fisher * (param - prev) ** 2
            penalty = term.sum() if penalty is None else penalty + term.sum()
        if penalty is None:
            return None
        return 0.5 * penalty

    def compute_importance(self, max_samples=None):
        """
        估计参数重要性：
        - ewc: 使用标签 cross-entropy 梯度平方（Fisher 近似）
        - mas: 使用输出 L2 范数梯度平方（不依赖标签）
        数据来源：累计测试集（覆盖已学任务的类别），避免只看当前任务。
        返回 (importance_dict or None, processed_samples)。
        """
        if max_samples is None:
            max_samples = self.ewc_samples_per_task
        if max_samples is None or max_samples <= 0:
            return None, 0
        stats = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.net.module.named_parameters()
            if param.requires_grad
        }
        # RWalk 路径积分累积，从当前 anchor 开始
        if self.cl_reg_method == "rwalk" and self.rwalk_path_score is None:
            self.rwalk_path_score = {
                name: torch.zeros_like(param, device="cpu")
                for name, param in self.net.module.named_parameters()
                if param.requires_grad
            }
        was_training = self.net.training
        self.net.eval()
        processed = 0
        data_iter = iter(self.run_config.test_loader)
        for batch in data_iter:
            if self.cl_reg_method == "ewc":
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.net.zero_grad()
                output = self.net(images)
                loss = self.criterion(output, labels)
            else:  # MAS | rWalk
                images, _ = batch  # 忽略标签
                images = images.to(self.device, non_blocking=True)
                self.net.zero_grad()
                output = self.net(images)
                loss = (output.pow(2).sum(dim=1).mean()) * 0.5
                
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            if not torch.isfinite(loss):
                continue
            
            loss.backward()
            batch_size = images.size(0)
            processed += batch_size
            with torch.no_grad():
                for name, param in self.net.module.named_parameters():
                    if param.grad is None or (name not in stats):
                        continue
                    g = param.grad.detach()
                    if not torch.isfinite(g).all():
                        continue
                    stats[name] += (g ** 2).cpu() * batch_size
            if processed >= max_samples:
                break
        if processed > 0:
            for name in stats:
                stats[name] /= float(processed)
        self.net.train(mode=was_training)
        return (stats if processed > 0 else None), processed

    def consolidate_ewc(self, fisher_dict, update_prev_params=True):
        """累积 Fisher；可选择是否更新参考参数（默认更新，仅在任务结束后调用）。"""
        if fisher_dict is None:
            return
        
        current_params = {
            name: param.detach().cpu().clone()
            for name, param in self.net.module.named_parameters()
            if param.requires_grad
        }
        
        if self.ewc_prev_params is None or self.ewc_fisher is None:
            self.ewc_prev_params = current_params
            clipped = {}
            for k, v in fisher_dict.items():
                if self.cl_reg_clip is not None:
                    clipped[k] = torch.clamp(v, max=self.cl_reg_clip)
                else:
                    clipped[k] = v.clone()
            self.ewc_fisher = clipped
            if self.cl_reg_method == "rwalk" and self.rwalk_path_score is None:
                self.rwalk_path_score = {
                    k: torch.zeros_like(v) for k, v in fisher_dict.items()
                }
            return
        
        if update_prev_params:
            self.ewc_prev_params = current_params
            
        for name, value in fisher_dict.items():
            if name in self.ewc_fisher:
                updated = value
                if self.cl_reg_clip is not None:
                    updated = torch.clamp(updated, max=self.cl_reg_clip)
                decay = self.cl_reg_decay if self.cl_reg_decay is not None else 1.0
                self.ewc_fisher[name] = self.ewc_fisher[name] * decay + updated
            else:
                if self.cl_reg_clip is not None:
                    self.ewc_fisher[name] = torch.clamp(value, max=self.cl_reg_clip)
                else:
                    self.ewc_fisher[name] = value.clone()
                    
        if self.cl_reg_method == "rwalk" and self.rwalk_path_score is None:
            self.rwalk_path_score = {
                k: torch.zeros_like(v) for k, v in fisher_dict.items()
            }
            
                
    def load_ortho_state(self, state):
        """加载旧任务的正交参考梯度方向，state=None 时重置。
        会自动跳过当前网络中不存在或 shape 不匹配的参数（兼容不同 backbone）。"""
        if not state:
            self.ortho_ref_grads = None
            self.prev_kd_grads = None
            self.prev_kd_grads_history = []
            return
        grads = state.get("grads", None)
        kd_prev = state.get("kd_prev_grads", None)
        if grads is None:
            self.ortho_ref_grads = None
        else:
            # 只保留当前模型中存在且 shape 匹配的参数
            cur_params = dict(self.net.module.named_parameters())
            filtered = {}
            for name, g in grads.items():
                if name in cur_params and cur_params[name].shape == g.shape:
                    filtered[name] = g.clone()
            self.ortho_ref_grads = filtered if filtered else None

        # 处理 kd_prev_grad_ortho 的参考梯度
        if kd_prev is not None:
            cur_params = dict(self.net.module.named_parameters())
            filtered_kd = {}
            for name, g in kd_prev.items():
                if name in cur_params and cur_params[name].shape == g.shape:
                    filtered_kd[name] = g.clone()
            self.prev_kd_grads = filtered_kd if filtered_kd else None
            self.prev_kd_grads_history = [self.prev_kd_grads] if self.prev_kd_grads is not None else []
        else:
            self.prev_kd_grads = None
            self.prev_kd_grads_history = []


    def export_ortho_state(self):
        """导出当前累积的正交参考梯度方向."""
        if self.ortho_ref_grads is None and self.prev_kd_grads is None:
            return None
        state = {}
        if self.ortho_ref_grads is not None:
            state["grads"] = {k: v.clone() for k, v in self.ortho_ref_grads.items()}
        if self.prev_kd_grads is not None:
            state["kd_prev_grads"] = {k: v.clone() for k, v in self.prev_kd_grads.items()}
        # 为避免膨胀，不序列化 history，history 由当前任务内维护
        return state

    def compute_ortho_reference(self, max_samples=None):
        """基于当前任务数据估计参数梯度方向，用于之后任务的正交投影。不需要旧任务数据，只在当前任务结束前调用一次即可。"""
        if max_samples is None or max_samples <= 0:
            max_samples = self.ortho_samples_per_task
        if max_samples <= 0:
            return None, 0
        if self.cl_ortho_method == "kd_ortho" and self.teacher_model is None:
            print("[Ortho] kd_ortho 启用但 teacher_model 为空，跳过参考梯度估计")
            return None, 0

        self.net.train()
        device = self.device
        grad_acc = {}
        processed = 0

        for images, labels in self.run_config.train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            self.net.zero_grad()
            output = self.net(images)
            # KD logits 引导的参考梯度（无需旧数据）
            if self.cl_ortho_method == "kd_ortho" and self.teacher_model is not None:
                with torch.no_grad():
                    t_out = self.teacher_model(images)
                temp = max(self.cl_kd_temperature, 1e-4)
                student_logp = F.log_softmax(output / temp, dim=1)
                teacher_prob = F.softmax(t_out / temp, dim=1)
                loss = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (temp * temp)
            elif self.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(
                    output, labels, self.run_config.label_smoothing
                )
            else:
                loss = self.criterion(output, labels)
            loss.backward()

            bsz = images.size(0)
            for name, p in self.net.module.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().cpu()
                if name not in grad_acc:
                    grad_acc[name] = g * bsz
                else:
                    grad_acc[name] += g * bsz

            processed += bsz
            if processed >= max_samples:
                break

        if processed == 0:
            return None, 0

        # 平均并可选与已有的 ortho_ref_grads 做衰减累积
        for name in grad_acc:
            grad_acc[name] /= float(processed)

        if self.ortho_ref_grads is None:
            # 直接使用当前任务的平均梯度方向
            self.ortho_ref_grads = grad_acc
        else:
            # 类似 EWC 的衰减融合，复用 cl_reg_decay 控制“记旧 vs 记新”
            decay = self.cl_reg_decay if self.cl_reg_decay is not None else 1.0
            merged = {}
            for name, g_new in grad_acc.items():
                if name in self.ortho_ref_grads:
                    g_old = self.ortho_ref_grads[name]
                    merged[name] = decay * g_old + (1.0 - decay) * g_new
                else:
                    merged[name] = g_new
            # 保留旧字典中没有被覆盖的键
            for name, g_old in self.ortho_ref_grads.items():
                if name not in merged:
                    merged[name] = g_old
            self.ortho_ref_grads = merged

        return self.ortho_ref_grads, processed

    def build_kd_ortho_reference(self, max_samples=None):
        """在任务开始时，用 KD (teacher vs student) 的梯度估计一个平均 g_old，供后续所有 step 投影使用。"""
        if max_samples is None or max_samples <= 0:
            max_samples = self.ortho_samples_per_task
        if max_samples <= 0 or self.teacher_model is None:
            return None
        self.net.train()
        device = self.device
        grad_acc = {}
        processed = 0
        for images, _labels in self.run_config.train_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_out = self.teacher_model(images)
                if not torch.isfinite(teacher_out).all():
                    teacher_out = torch.nan_to_num(teacher_out, nan=0.0, posinf=1e4, neginf=-1e4)
            self.net.zero_grad()
            student_out = self.net(images)
            if not torch.isfinite(student_out).all():
                student_out = torch.nan_to_num(student_out, nan=0.0, posinf=1e4, neginf=-1e4)
            T = self.cl_kd_temperature if self.cl_kd_temperature > 0 else 1.0
            student_logp = F.log_softmax(student_out / T, dim=1)
            teacher_prob = F.softmax(teacher_out / T, dim=1)
            kd_loss = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (T * T)
            kd_loss.backward()

            bsz = images.size(0)
            for name, p in self.net.module.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().cpu()
                if name not in grad_acc:
                    grad_acc[name] = g * bsz
                else:
                    grad_acc[name] += g * bsz

            processed += bsz
            if processed >= max_samples:
                break

        if processed == 0:
            return None
        for name in grad_acc:
            grad_acc[name] /= float(processed)
        return grad_acc

        
    def set_teacher(self, teacher_model):
        """设置上一任务的 teacher 模型，用于 KD / 正交更新."""
        if teacher_model is None:
            self.teacher_model = None
            return
        # 冻结参数，不参与梯度
        for p in teacher_model.parameters():
            p.requires_grad = False
        if torch.cuda.is_available():
            if not isinstance(teacher_model, torch.nn.DataParallel):
                teacher_model = torch.nn.DataParallel(teacher_model)
            teacher_model = teacher_model.to(self.device)
        teacher_model.eval()
        self.teacher_model = teacher_model


    @property
    def save_path(self):
        if self._save_path is None:
            # 将 task_id 纳入保存目录，最小化对代码其他部分的侵入
            save_path = os.path.join(self.path, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, "logs")
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    """ net info """

    # noinspection PyUnresolvedReferences
    def net_flops(self):
        data_shape = [1] + list(self.run_config.data_provider.data_shape)

        if isinstance(self.net, nn.DataParallel):
            net = self.net.module
        else:
            net = self.net
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop

    def cmp_lat(self, batch_size=64, iterations=10):
        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)
        with torch.no_grad():
            images = torch.zeros(data_shape)
            # GPU-latency
            images.cuda()
            print("=========GPU Speed Testing=========")
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                output = self.net(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            gpu_latency = elapsed_time_ms / iterations
            torch.cuda.empty_cache()
            print("=========CPU Speed Testing=========")
            images.cpu()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                output = self.net_on_cpu_for_latency(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            cpu_latency = elapsed_time_ms / iterations
            torch.cuda.empty_cache()
            print("GPU latency:", gpu_latency, "CPU latency:", cpu_latency)

    # noinspection PyUnresolvedReferences
    def net_latency(self, l_type="gpu4", fast=True, given_net=None):
        if "gpu" in l_type:
            l_type, batch_size = l_type[:3], int(l_type[3:])
        else:
            batch_size = 1

        data_shape = [batch_size] + list(self.run_config.data_provider.data_shape)

        if given_net is not None:
            net = given_net
        else:
            net = self.net.module  # 并行后的修改
            # net = self.net

        if l_type == "mobile":
            predicted_latency = 0
            try:
                assert isinstance(net, ProxylessNASNets)
                # first conv
                predicted_latency += self.latency_estimator.predict(
                    "Conv", [224, 224, 3], [112, 112, net.first_conv.out_channels]
                )
                # feature mix layer
                predicted_latency += self.latency_estimator.predict(
                    "Conv_1",
                    [7, 7, net.feature_mix_layer.in_channels],
                    [7, 7, net.feature_mix_layer.out_channels],
                )
                # classifier
                predicted_latency += self.latency_estimator.predict(
                    "Logits",
                    [7, 7, net.classifier.in_features],
                    [net.classifier.out_features],  # 1000
                )
                # blocks
                fsize = 112
                for block in net.blocks:
                    mb_conv = block.mobile_inverted_conv
                    shortcut = block.shortcut
                    if isinstance(mb_conv, MixedEdge):
                        mb_conv = mb_conv.active_op
                    if isinstance(shortcut, MixedEdge):
                        shortcut = shortcut.active_op

                    if mb_conv.is_zero_layer():
                        continue
                    if shortcut is None or shortcut.is_zero_layer():
                        idskip = 0
                    else:
                        idskip = 1
                    out_fz = fsize // mb_conv.stride
                    block_latency = self.latency_estimator.predict(
                        "expanded_conv",
                        [fsize, fsize, mb_conv.in_channels],
                        [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio,
                        kernel=mb_conv.kernel_size,
                        stride=mb_conv.stride,
                        idskip=idskip,
                    )
                    predicted_latency += block_latency
                    fsize = out_fz
            except Exception:
                predicted_latency = 200
                print("fail to predict the mobile latency")
            return predicted_latency, None
        elif l_type == "cpu":
            if fast:
                n_warmup = 1
                n_sample = 2
            else:
                n_warmup = 10
                n_sample = 100
            try:
                self.net_on_cpu_for_latency.set_active_via_net(net)
            except AttributeError:
                print(
                    type(self.net_on_cpu_for_latency),
                    " do not `support set_active_via_net()`",
                )
            net = self.net_on_cpu_for_latency
            images = torch.zeros(data_shape, device=torch.device("cpu"))
        elif l_type == "gpu":
            if fast:
                n_warmup = 5
                n_sample = 10
            else:
                n_warmup = 50
                n_sample = 100
            images = torch.zeros(data_shape, device=self.device)
        else:
            # raise NotImplementedError
            return 0, 0  # latency = 0 when Hardware==None

        measured_latency = {"warmup": [], "sample": []}
        net.eval()
        with torch.no_grad():
            for i in range(n_warmup + n_sample):
                start_time = time.time()
                net(images)
                used_time = (time.time() - start_time) * 1e3  # ms
                if i >= n_warmup:
                    measured_latency["sample"].append(used_time)
                else:
                    measured_latency["warmup"].append(used_time)
        net.train()
        return sum(measured_latency["sample"]) / n_sample, measured_latency

    def print_net_info(self, measure_latency=None):
        # print(self.net)
        # parameters
        if isinstance(self.net, nn.DataParallel):
            total_params = count_parameters(self.net.module)
        else:
            total_params = count_parameters(self.net)
        if self.out_log:
            print("Total training params: %.2fM" % (total_params / 1e6))
        net_info = {
            "param": "%.2fM" % (total_params / 1e6),
        }

        # flops
        flops = self.net_flops()
        if self.out_log:
            print("Total FLOPs: %.1fM" % (flops / 1e6))
        net_info["flops"] = "%.1fM" % (flops / 1e6)

        # latency
        print("Measure latency for types:", measure_latency)
        latency_types = [] if measure_latency is None else measure_latency.split("#")
        for l_type in latency_types:
            latency, measured_latency = self.net_latency(
                l_type, fast=False, given_net=None
            )
            if self.out_log:
                print("Estimated %s latency: %.3fms" % (l_type, latency))
            net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}
        with open("%s/net_info.txt" % self.logs_path, "w") as fout:
            fout.write(json.dumps(net_info, indent=4) + "\n")

    """ save and load models """

    def save_model(
        self, checkpoint=None, is_best=False, model_name=None, HardwareClass=None
    ):
        if checkpoint is None:
            checkpoint = {"state_dict": self.net.module.state_dict()}

        if model_name is None:
            model_name = "checkpoint.pth.tar"

        checkpoint["dataset"] = (
            self.run_config.dataset
        )  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, "latest.txt")
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, "w") as fout:
            fout.write(model_path + "\n")
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, "model_best.pth.tar")
            torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

        # --------- 简单清理：仅保留最近若干个 checkpoint，避免线性膨胀 ---------
        try:
            max_keep = getattr(self, "max_keep_checkpoints", 5)
            keep_names = {
                os.path.abspath(model_path),
                os.path.abspath(os.path.join(self.save_path, "model_best.pth.tar")),
                os.path.abspath(os.path.join(self.save_path, "warmup.pth.tar")),
                os.path.abspath(os.path.join(self.save_path, "global.pth.tar")),
            }
            # 按修改时间排序
            all_ckpts = [
                os.path.join(self.save_path, f)
                for f in os.listdir(self.save_path)
                if f.endswith(".pth.tar")
            ]
            all_ckpts = sorted(all_ckpts, key=lambda p: os.path.getmtime(p), reverse=True)
            survivors = []
            for p in all_ckpts:
                ap = os.path.abspath(p)
                if ap in keep_names:
                    survivors.append(ap)
                    continue
                if len(survivors) < max_keep:
                    survivors.append(ap)
                else:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        except Exception:
            # 清理失败不影响训练主流程
            pass

    def load_clients_opt(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline().strip()
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = os.path.join(self.save_path, "checkpoint.pth.tar")
            with open(latest_fname, "w") as fout:
                fout.write(model_fname + "\n")
        print(f"=> loading client checkpoint '{model_fname}'")

        checkpoint = torch.load(
            model_fname, map_location=torch.device("cpu")
        )  # 加载到 CPU
        client_key = f"{self.run_config.data_provider.client_id}_weight_optimizer"
        if client_key in checkpoint:
            self.optimizer.load_state_dict(checkpoint[client_key])  # 加载优化器状态
        print(f"=> loaded client checkpoint '{model_fname}'")

    def return_model_dict(
        self,
    ):
        _w = self.net.module.state_dict()
        return _w

    def get_run_manager_model(self):
        return self.net.module

    def get_local_data_weight(self):
        return self.run_config.data_provider.trn_set_length

    def load_model(self, model_fname=None, HardwareClass=None):
        latest_fname = os.path.join(self.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline().strip()
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = os.path.join(self.save_path, "checkpoint.pth.tar")
                with open(latest_fname, "w") as fout:
                    fout.write(model_fname + "\n")
            if self.out_log:
                print(f"=> loading checkpoint '{model_fname}'")

            checkpoint = torch.load(model_fname, map_location="cpu")
            self.net.module.load_state_dict(checkpoint["state_dict"])  # 加载模型权重
            if "round" in checkpoint:
                self.round = checkpoint["round"] + 1
            if self.out_log:
                print(f"=> loaded checkpoint '{model_fname}'")
        except Exception as e:
            print("Exception about load model:", e)
            if self.out_log:
                print(f"Failed to load checkpoint from {self.save_path}")

    def save_config(self, print_info=True):
        """dump run_config and net_config to the model_folder"""
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, "net.config")
        json.dump(
            self.net.module.config,
            open(net_save_path, "w"),
            indent=4,
            default=set_to_list,
        )  # 并行后的修改

        if print_info:
            print("Network configs dump to %s" % net_save_path)

        run_save_path = os.path.join(self.path, "run.config")
        json.dump(
            self.run_config.config,
            open(run_save_path, "w"),
            indent=4,
            default=set_to_list,
        )
        if print_info:
            print("Run configs dump to %s" % run_save_path)

    """ train and test """
    def write_log(self, log_str, prefix, should_print=True):
        """prefix: valid, test"""
        if prefix in ["valid", "test"]:
            with open(os.path.join(self.logs_path, "test_console.txt"), "a") as fout:
                fout.write(log_str + "\n")
                fout.flush()
        # if prefix in ["valid", "test", "train"]:
        #     with open(os.path.join(self.logs_path, "train_console.txt"), "a") as fout:
        #         if prefix in ["valid", "test"]:
        #             fout.write("=" * 10)
        #         fout.write(log_str + "\n")
        #         fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if is_test:
            data_loader = self.run_config.test_loader
            print("USE: test_loader")
        else:
            data_loader = self.run_config.valid_loader
        data_loader = list(data_loader)
        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()

        net = net.cuda()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        per_task_total = {}
        per_task_correct = {}
        class_to_task = None
        if is_test:
            fm = getattr(self.run_config.data_provider, "fcl_manager", None)
            if fm is not None and hasattr(fm, "_task_classes"):
                class_to_task = {}
                for t, cls_list in fm._task_classes.items():
                    for c in cls_list:
                        class_to_task[int(c)] = int(t)
                # 打印当前任务划分的类别信息，便于对齐精度
                cls_str = "; ".join(
                    [f"T{t}: {sorted([int(c) for c in fm._task_classes[t]])}" for t in sorted(fm._task_classes.keys())]
                )
                self.write_log(f"task_class_split: {cls_str}", prefix="test", should_print=True)
                
        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(
                    self.device, non_blocking=True
                )
                # 跳过包含非有限值的 batch
                if not torch.isfinite(images).all():
                    continue
                # 9.to(device,non_blocking=True)
                # compute output
                output = net(images)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                loss = self.criterion(output, labels)
                if not torch.isfinite(loss):
                    continue
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                if class_to_task is not None:
                    preds = output.argmax(dim=1)
                    for idx in range(labels.size(0)):
                        t = class_to_task.get(int(labels[idx].item()))
                        if t is None:
                            continue
                        per_task_total[t] = per_task_total.get(t, 0) + 1
                        if int(preds[idx].item()) == int(labels[idx].item()):
                            per_task_correct[t] = per_task_correct.get(t, 0) + 1
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        if is_test and per_task_total:
            self.last_task_acc = {
                t: (per_task_correct.get(t, 0) * 100.0 / per_task_total[t])
                for t in per_task_total
            }
            log_str = "per_task_test_top1: " + ", ".join(
                [f"T{t}:{acc:.2f}" for t, acc in sorted(self.last_task_acc.items())]
            )
            # 结果随着search、retrain阶段的变化而变化，打印到test_console.txt中
            self.write_log(log_str, prefix="test", should_print=True)
        else:
            self.last_task_acc = None

        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def train_run_manager_one_epoch(self, adjust_lr_func, prev_params_snapshot=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        lr = AverageMeter()
        # RWalk 路径积分需要的累计表
        if self.cl_reg_method == "rwalk" and self.rwalk_path_score is None:
            self.rwalk_path_score = {
                name: torch.zeros_like(param, device="cpu")
                for name, param in self.net.module.named_parameters()
                if param.requires_grad
            }
            
        # 预估正交参考：
        # - kd_ortho：任务开始时预估一个全局 g_old 参考
        # - 其他 ortho：若还没有 ortho_ref_grads，尝试基于当前任务数据估计
        # - prev_grad_ortho / kd_prev_grad_ortho：参考来自上一轮梯度，训练过程中动态更新，无需预估
        if (
            self.cl_ortho_method == "kd_ortho"
            and self.teacher_model is not None
            and self.cl_kd_logit_lambda > 0
        ):
            self.kd_ortho_ref_grads = self.build_kd_ortho_reference(
                max_samples=self.ortho_samples_per_task
            )
        elif (
            self.cl_ortho_method != "none"
            and self.cl_ortho_method != "kd_ortho"
            and self.ortho_ref_grads is None
            and self.ortho_samples_per_task is not None
            and self.ortho_samples_per_task > 0
        ):
            ortho_ref, processed = self.compute_ortho_reference(
                max_samples=self.ortho_samples_per_task
            )
            if ortho_ref is not None:
                print(
                    f"[Ortho] precomputed ortho_ref_grads for method={self.cl_ortho_method}, processed={processed}"
                )
        # kd_prev_grad_ortho：参考期望来自上一任务保存的 kd_prev_grads，不再用当前任务数据预估
        # 若没有加载到有效的 prev_kd_grads，则后续训练中将跳过投影

        # switch to train mode
        self.net.train()

        end = time.time()
        ewc_meter = AverageMeter()

        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device, non_blocking=True), labels.to(
                self.device, non_blocking=True
            )

            # ========== 前向 ==========
            output = self.net(images)
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

            # CE 基础 loss
            if self.run_config.label_smoothing > 0:
                ce_loss = cross_entropy_with_label_smoothing(
                    output, labels, self.run_config.label_smoothing
                )
            else:
                ce_loss = self.criterion(output, labels)

            if not torch.isfinite(ce_loss):
                if i == 0 or i % 10 == 0:
                    print(
                        f"[Train] non-finite ce_loss detected (loss={ce_loss.item()}), clamp to 0 at batch {i}"
                    )
                ce_loss = torch.zeros_like(ce_loss)

            # EWC / MAS / RWalk 正则项
            ewc_penalty = self._ewc_penalty()
            if ewc_penalty is not None and torch.isfinite(ewc_penalty):
                ewc_meter.update(ewc_penalty.item(), images.size(0))

            # ========== KD logit loss（只算数值，真正用在哪取决于模式）==========
            kd_loss = None
            if (
                self.teacher_model is not None
                and self.cl_kd_logit_lambda > 0
                and self.cl_kd_method in ["logit", "logit_conf"]
            ):
                with torch.no_grad():
                    teacher_out = self.teacher_model(images)
                    if not torch.isfinite(teacher_out).all():
                        teacher_out = torch.nan_to_num(
                            teacher_out, nan=0.0, posinf=1e4, neginf=-1e4
                        )
                T = self.cl_kd_temperature if self.cl_kd_temperature > 0 else 1.0
                student_log_probs = F.log_softmax(output / T, dim=1)
                teacher_probs = F.softmax(teacher_out / T, dim=1)

                if self.cl_kd_method == "logit":
                    kd_loss = F.kl_div(
                        student_log_probs, teacher_probs, reduction="batchmean"
                    ) * (T * T)
                else:  # logit_conf
                    # 只保留 teacher 置信度高于阈值的类别做 KL
                    mask = (teacher_probs > self.cl_kd_conf_threshold).float()
                    if mask.sum() > 0:
                        log_teacher = torch.log(teacher_probs + 1e-12)
                        kl_elem = (teacher_probs * (log_teacher - student_log_probs)) * mask
                        kd_loss = (kl_elem.sum() / mask.sum()) * (T * T)
                    else:
                        kd_loss = None

            # ========== kd_ortho 模式 / 其余ortho 方法 ==========
            # ========== 统一 total_loss 计算 ==========
            total_loss = ce_loss
            if kd_loss is not None:
                total_loss = total_loss + self.cl_kd_logit_lambda * kd_loss
            if ewc_penalty is not None and torch.isfinite(ewc_penalty):
                penalty_term = self.ewc_lambda * ewc_penalty
                if self.cl_penalty_clip is not None:
                    penalty_term = torch.clamp(
                        penalty_term, max=self.cl_penalty_clip
                    )
                total_loss = total_loss + penalty_term

            # ========== kd_ortho 特殊逻辑 ==========
            cos_before, cos_after = None, None
            if (
                self.cl_ortho_method == "kd_ortho"
                and self.teacher_model is not None
                and self.cl_kd_logit_lambda > 0
                and kd_loss is not None
            ):
                # 1) 准备 g_old：优先使用任务级缓存，否则按当前 batch 估计
                old_grads = None
                if self.kd_ortho_ref_grads is not None:
                    old_grads = {
                        name: g.to(self.device) for name, g in self.kd_ortho_ref_grads.items()
                    }
                else:
                    self.optimizer.zero_grad()
                    self.net.zero_grad()
                    kd_loss.backward(retain_graph=True)
                    old_grads = {}
                    for name, p in self.net.module.named_parameters():
                        if p.grad is not None and p.requires_grad:
                            old_grads[name] = p.grad.detach().clone()
                    self.optimizer.zero_grad()
                    self.net.zero_grad()

                # 2) 反向计算当前梯度
                self.optimizer.zero_grad()
                self.net.zero_grad()
                total_loss.backward()

                # 3) 投影到 g_old 的正交子空间
                eps = 1e-12
                g_list_before = []
                g_old_list = []
                for name, p in self.net.module.named_parameters():
                    if p.grad is None or name not in old_grads:
                        continue
                    g = p.grad
                    g_old = old_grads[name].to(g.device)

                    g_flat = g.view(-1)
                    g_old_flat = g_old.view(-1)
                    denom = torch.dot(g_old_flat, g_old_flat) + eps
                    if denom.item() == 0.0:
                        continue

                    dot = torch.dot(g_flat, g_old_flat)
                    proj_coef = dot / denom

                    g_list_before.append(g_flat.detach().clone())
                    g_old_list.append(g_old_flat.detach().clone())

                    scale = self.cl_ortho_scale
                    g_ortho_flat = g_flat - scale * proj_coef * g_old_flat
                    p.grad.copy_(g_ortho_flat.view_as(p))

                if g_list_before and g_old_list:
                    g_flat_all = torch.cat(g_list_before)
                    g_old_all = torch.cat(g_old_list)
                    cos_before = F.cosine_similarity(
                        g_flat_all.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
                    ).item()
                    g_list_after = []
                    for name, p in self.net.module.named_parameters():
                        if p.grad is None or name not in old_grads:
                            continue
                        g_list_after.append(p.grad.detach().view(-1))
                    if g_list_after:
                        g_flat_after = torch.cat(g_list_after)
                        cos_after = F.cosine_similarity(
                            g_flat_after.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
                        ).item()

            elif self.cl_ortho_method == "prev_grad_ortho":
                # 使用上一轮梯度做正交投影
                self.optimizer.zero_grad()
                self.net.zero_grad()
                total_loss.backward()
                cos_before, cos_after = self.apply_ortho_projection_with_previous_gradients(return_cos=True, use_kd=False)
                # 本轮梯度投影后立即保存，供下一轮使用
                self.save_gradients()

            elif (
                self.cl_ortho_method == "kd_prev_grad_ortho"
                and self.teacher_model is not None
            ):
                # 使用上一任务保存的 KD 参考梯度（固定 teacher），不随当前 batch 更新；如果加载失败，则跳过投影
                self.optimizer.zero_grad()
                self.net.zero_grad()
                total_loss.backward()
                cos_before, cos_after = (None, None)
                if self.prev_kd_grads is not None:
                    cos_before, cos_after = self.apply_ortho_projection_with_previous_gradients(
                        return_cos=True, use_kd=True, use_history=False
                    )
                
            else:
                # ========== 其他 ortho / 普通模式 ==========
                self.optimizer.zero_grad()
                self.net.zero_grad()
                total_loss.backward()
                if self.cl_ortho_method != "none":
                    cos_before, cos_after = self.apply_ortho_projection(return_cos=True)

            # 更新参数
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            self.optimizer.step()
            # RWalk 路径积分累积
            if (
                self.cl_reg_method == "rwalk"
                and prev_params_snapshot is not None
                and self.rwalk_path_score is not None
            ):
                with torch.no_grad():
                    for name, param in self.net.module.named_parameters():
                        if name not in prev_params_snapshot:
                            continue
                        delta = (param.detach().cpu() - prev_params_snapshot[name])
                        self.rwalk_path_score[name] += delta.abs()
                        prev_params_snapshot[name] = param.detach().cpu().clone()

            # 打印正交余弦信息（仅当有投影时）
            if (
                self.cl_ortho_method != "none"
                and cos_before is not None
                and cos_after is not None
                and (i % 50 == 0)
            ):
                print(f"[{self.cl_ortho_method}] batch {i}, cos(before)={cos_before:.4f}, cos(after)={cos_after:.4f}")

            # ========== 统计&时间 ==========
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(total_loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            lr.update(new_lr, 1)

            batch_time.update(time.time() - end)
            end = time.time()

        return top1, top5, losses, lr, ewc_meter
    
    def apply_ortho_projection(self, return_cos: bool = False):
        """
        在 optimizer.step() 之前，对当前梯度做一次
        g <- g - alpha * proj_{g_old}(g)，
        其中 g_old 来自 self.ortho_ref_grads。
        """
        if (
            self.cl_ortho_method == "none"
            or self.cl_ortho_scale <= 0
            or self.ortho_ref_grads is None
        ):
            return (None, None) if return_cos else None

        use_fisher = self.ewc_fisher is not None
        g_list_before, g_old_list, g_list_after = [], [], []

        # 日志：标记本轮投影使用的是 Fisher 加权还是普通 dot
        if not hasattr(self, "_ortho_proj_log_flag"):
            self._ortho_proj_log_flag = {}
        log_key = "fisher" if use_fisher else "dot"
        if not self._ortho_proj_log_flag.get(log_key, False):
            print(f"[Ortho] apply_ortho_projection using {'Fisher-weighted' if use_fisher else 'dot'} metric (method={self.cl_ortho_method})")
            self._ortho_proj_log_flag[log_key] = True

        with torch.no_grad():
            for name, p in self.net.module.named_parameters():
                if p.grad is None:
                    continue
                if name not in self.ortho_ref_grads:
                    continue

                g = p.grad
                g_old = self.ortho_ref_grads[name].to(g.device)

                g_flat = g.view(-1)
                g_old_flat = g_old.view(-1)

                fisher_flat = None
                if use_fisher and name in self.ewc_fisher:
                    fisher_flat = self.ewc_fisher[name].to(g.device).view(-1)

                # Fisher 度量：<g, g_old>_F = Σ F * g * g_old
                if fisher_flat is not None:
                    dot = (fisher_flat * g_flat * g_old_flat).sum()
                    denom = (fisher_flat * g_old_flat * g_old_flat).sum() + 1e-12
                else:
                    dot = g_flat.dot(g_old_flat)
                    denom = g_old_flat.dot(g_old_flat) + 1e-12

                if denom.item() == 0.0:
                    continue

                proj = dot / denom

                # PCGrad：只在冲突（dot<0）时投影；OGD/kd_ortho：无条件投影
                if self.cl_ortho_method == "pcgrad" and dot >= 0:
                    continue

                if return_cos:
                    g_list_before.append(g_flat.detach().clone())
                    g_old_list.append(g_old_flat.detach().clone())

                scale = self.cl_ortho_scale
                g_perp_flat = g_flat - scale * proj * g_old_flat
                p.grad = g_perp_flat.view_as(p)
                if return_cos:
                    g_list_after.append(p.grad.detach().view(-1))

        if return_cos and g_list_before and g_old_list:
            g_flat_all = torch.cat(g_list_before)
            # g_old_list 与 g_list_before 等长；每个元素已是聚合方向
            g_old_all = torch.cat(g_old_list)
            if g_flat_all.numel() == g_old_all.numel():
                cos_before = F.cosine_similarity(
                    g_flat_all.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
                ).item()
            else:
                cos_before = None
            cos_after = None
            if g_list_after:
                g_flat_after = torch.cat(g_list_after)
                if g_flat_after.numel() == g_old_all.numel():
                    cos_after = F.cosine_similarity(
                        g_flat_after.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
                    ).item()
            return cos_before, cos_after
        return (None, None) if return_cos else None

    def apply_ortho_projection_with_previous_gradients(self, return_cos: bool = False, use_kd: bool = False, use_history: bool = False):
        """
        使用上一轮缓存的梯度对当前梯度做正交投影。
        use_kd=True 时使用 prev_kd_grads，否则使用 prev_grads。
        use_history 参数保留但当前实现不再叠加历史，仅使用已加载的参考。
        """
        if (
            self.cl_ortho_method not in ["prev_grad_ortho", "kd_prev_grad_ortho"]
            or self.cl_ortho_scale <= 0
        ):
            return (None, None) if return_cos else None

        ref_list = []
        if use_kd:
            if self.prev_kd_grads is not None:
                ref_list = [self.prev_kd_grads]
        else:
            if self.prev_grads is not None:
                ref_list = [self.prev_grads]

        if not ref_list:
            return (None, None) if return_cos else None

        g_list_before, g_old_list, g_list_after = [], [], []
        eps = 1e-12
        with torch.no_grad():
            # 对每个参数，依次在历史参考上做投影，得到同时正交于所有参考的方向
            for name, p in self.net.module.named_parameters():
                if p.grad is None:
                    continue
                # 收集该参数的所有参考梯度
                refs = []
                for ref_grads in ref_list:
                    if name in ref_grads:
                        refs.append(ref_grads[name].to(p.device))
                if not refs:
                    continue
                g_flat = p.grad.view(-1)
                if return_cos:
                    g_list_before.append(g_flat.detach().clone())
                    # 记录一个聚合的参考方向，避免长度不匹配
                    g_old_agg = torch.stack([r.view(-1) for r in refs], dim=0).mean(dim=0)
                    g_old_list.append(g_old_agg.detach().clone())
                # 逐个参考投影（Gram-Schmidt 风格）
                for g_old in refs:
                    g_old_flat = g_old.view(-1)
                    denom = g_old_flat.dot(g_old_flat) + eps
                    if denom.item() == 0.0:
                        continue
                    dot = g_flat.dot(g_old_flat)
                    proj = dot / denom
                    g_flat = g_flat - self.cl_ortho_scale * proj * g_old_flat
                p.grad = g_flat.view_as(p)
                if return_cos:
                    g_list_after.append(p.grad.detach().view(-1))

        if return_cos and g_list_before and g_old_list:
            g_flat_all = torch.cat(g_list_before)
            g_old_all = torch.cat(g_old_list)
            cos_before = F.cosine_similarity(
                g_flat_all.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
            ).item()
            cos_after = None
            if g_list_after:
                g_flat_after = torch.cat(g_list_after)
                cos_after = F.cosine_similarity(
                    g_flat_after.unsqueeze(0), g_old_all.unsqueeze(0), dim=1
                ).item()
            return cos_before, cos_after
        return (None, None) if return_cos else None

    def save_gradients(self):
        """
        将当前梯度缓存到 self.prev_grads（CPU），供下次 prev_grad_ortho 使用。
        """
        grads = {}
        for name, p in self.net.module.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            grads[name] = p.grad.detach().cpu().clone()
        self.prev_grads = grads if grads else None

    def compute_kd_gradients(self, kd_loss, detach_cpu: bool = True):
        """
        基于 kd_loss 计算梯度，返回字典；retain_graph=True 不清理计算图。
        """
        if kd_loss is None:
            return None
        params = [(n, p) for n, p in self.net.module.named_parameters() if p.requires_grad]
        tensors = [p for _, p in params]
        grads_raw = torch.autograd.grad(
            kd_loss,
            tensors,
            retain_graph=True,
            allow_unused=True,
        )
        grads = {}
        for (name, _), g in zip(params, grads_raw):
            if g is None:
                continue
            grads[name] = g.detach().cpu().clone() if detach_cpu else g.detach()
        return grads if grads else None

    def _append_kd_history(self, grads_dict):
        """将新的 KD 参考梯度加入历史，控制长度。"""
        if grads_dict is None:
            return
        self.prev_kd_grads_history.append(grads_dict)
        if self.ortho_kd_history_max > 0 and len(self.prev_kd_grads_history) > self.ortho_kd_history_max:
            self.prev_kd_grads_history = self.prev_kd_grads_history[-self.ortho_kd_history_max:]


    def train_run_manager(
        self,
        start_local_epoch=None,
        last_local_epoch=None,
        print_top5=False,
        server_model=None,
        writer=None,
    ):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.module.load_state_dict(server_model.module.state_dict())
            else:
                self.net.module.load_state_dict(server_model.state_dict())

        nBatch = len(self.run_config.train_loader)

        start_epoch = start_local_epoch
        n_epochs = last_local_epoch
        val_loss, val_acc, val_acc5, lr = None, None, None, None
        train_acc_top1_arr, train_acc_top5_arr, train_losses_arr = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        # RWalk 路径快照
        prev_params_snapshot = None
        if self.cl_reg_method == "rwalk":
            prev_params_snapshot = {
                name: param.detach().cpu().clone()
                for name, param in self.net.module.named_parameters()
                if param.requires_grad
            }
        for epoch in range(start_epoch, n_epochs):
            train_acc_top1, train_acc_top5, train_losses, lr, ewc_meter = self.train_run_manager_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(
                    self.optimizer, epoch, i, nBatch
                ),
                prev_params_snapshot=prev_params_snapshot,
            )
            train_acc_top1_arr.update(train_acc_top1.avg)
            train_acc_top5_arr.update(train_acc_top5.avg)
            train_losses_arr.update(train_losses.avg)
            cid = str(self.run_config.data_provider.client_id)
            tb_prefix = f"task_{self.task_id}_client_{cid}_"
            writer.add_scalar(tb_prefix + "_trn_top1", train_acc_top1.avg, epoch)
            writer.add_scalar(tb_prefix + "_trn_top5", train_acc_top5.avg, epoch)
            writer.add_scalar(tb_prefix + "_trn_loss", train_losses.avg, epoch)
            if self.ewc_lambda > 0:
                writer.add_scalar(tb_prefix + "_ewc_penalty", ewc_meter.avg, epoch)

            if epoch % 3 == 0:
                val_loss, val_acc, val_acc5 = self.validate(
                    is_test=False, return_top5=True
                )
                writer.add_scalar(tb_prefix + "_val_loss", val_loss, epoch)
                writer.add_scalar(tb_prefix + "_val_top1", val_acc, epoch)
                writer.add_scalar(tb_prefix + "_val_top5", val_acc5, epoch)
        lr_value = lr.avg if isinstance(lr, AverageMeter) else lr
        return (
            train_losses_arr.avg,
            train_acc_top1_arr.avg,
            train_acc_top5_arr.avg,
            val_loss,
            val_acc,
            val_acc5,
            lr_value,
        )

    def inference_run_manager(self, server_model=None, is_test=True):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.module.load_state_dict(server_model.module.state_dict())
            else:
                self.net.module.load_state_dict(server_model.state_dict())
        else:
            print("WARNING: Have no server model")
        val_loss, val_top1, val_top5 = self.validate(is_test=True, return_top5=True)
        return val_loss, val_top1, val_top5

    print("CifarRunConfig初始化完成...")


"""
    CifarRunManager:cifar数据集的运行配置类
"""


class CifarRunConfig(RunConfig):

    def __init__(
        self,
        client_id=10,
        dataset_location=None,
        n_epochs=150,
        init_lr=0.05,
        lr_schedule_type="cosine",
        lr_schedule_param=None,
        dataset="cifar",
        train_batch_size=256,
        test_batch_size=500,
        valid_size=None,
        opt_type="sgd",
        opt_param=None,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys="bn",
        model_init="he_fout",
        init_div_groups=False,
        validation_frequency=1,
        print_frequency=10,
        n_worker=8,
        search=True,
        task_id=1,
        is_client:bool=True,
        ewc_lambda: float = 0.0,
        ewc_samples_per_task: int = 0,
        ewc_online_interval: int = 0,
        cl_reg_method: str = "mas",
        cl_reg_decay: float = 1.0,
        cl_reg_clip: float = None,
        cl_penalty_clip: float = None,
        cl_kd_method: str = "none",
        cl_kd_logit_lambda: float = 0.0,
        cl_kd_temperature: float = 2.0,
        cl_kd_conf_threshold: float = 0.5,
        cl_ortho_method: str = "none",
        cl_ortho_scale: float = 1.0,
        ortho_samples_per_task: int = 2048,
        **kwargs,
    ):
        print(f"CifarRunConfig初始化开始... is_client={is_client} client_id={client_id}")
        # 兼容主程序里传入的联邦/持续学习控制参数
        num_users = kwargs.get("num_users", 10)
        num_tasks = kwargs.get("num_tasks", 10)
        explicit_cpt = kwargs.get("classes_per_task", None)

        super(CifarRunConfig, self).__init__(
            client_id,
            dataset_location,
            n_epochs,
            init_lr,
            lr_schedule_type,
            lr_schedule_param,
            dataset,
            train_batch_size,
            test_batch_size,
            valid_size,
            opt_type,
            opt_param,
            weight_decay,
            label_smoothing,
            no_decay_keys,
            model_init,
            init_div_groups,
            validation_frequency,
            print_frequency,
            search,
            task_id,
            is_client
        )
        self.dataset_location = dataset_location
        self.n_worker = n_worker
        self.num_clients = int(num_users)
        self.num_tasks = int(num_tasks)
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_samples_per_task = int(ewc_samples_per_task)
        self.ewc_online_interval = int(ewc_online_interval)
        self.cl_reg_method = str(cl_reg_method).lower()
        self.cl_reg_decay = float(cl_reg_decay)
        self.cl_reg_clip = cl_reg_clip
        self.cl_penalty_clip = cl_penalty_clip
        
        # KD 相关
        self.cl_kd_method = str(cl_kd_method).lower()
        self.cl_kd_logit_lambda = float(cl_kd_logit_lambda)
        self.cl_kd_temperature = float(cl_kd_temperature)
        self.cl_kd_conf_threshold = float(cl_kd_conf_threshold)
        
        # 正交更新相关
        self.cl_ortho_method = str(cl_ortho_method).lower()
        self.cl_ortho_scale = float(cl_ortho_scale)
        self.ortho_samples_per_task = int(ortho_samples_per_task)

        # 推导每任务类别数（若未显式给出）
        total_classes = None
        ds_lower = str(self.dataset).lower()
        if "cifar100" in ds_lower:
            total_classes = 100
        elif "cifar10" in ds_lower or "cifar" == ds_lower:
            total_classes = 10

        if explicit_cpt is not None:
            self.classes_per_task = int(explicit_cpt)
        elif total_classes is not None:
            if self.num_tasks <= 0 or total_classes % self.num_tasks != 0:
                raise ValueError(f"无法将 {total_classes} 个类别均分到 {self.num_tasks} 个任务，请检查 num_tasks 参数")
            self.classes_per_task = total_classes // self.num_tasks
        else:
            self.classes_per_task = None

    @property
    def data_config(self):
        return {
            "client_id": self.client_id,
            "task_id": self.task_id,
            "dataset_location": self.dataset_location,
            "train_batch_size": self.train_batch_size,
            "test_batch_size": self.test_batch_size,
            # "valid_size": self.valid_size,
            "n_worker": self.n_worker,
            "is_client": self.is_client,
            "num_clients": self.num_clients,
            "num_tasks": self.num_tasks,
            "classes_per_task": self.classes_per_task,
            "search": self.search,
        }

    print("CifarRunConfig初始化完成...")


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
