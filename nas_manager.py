# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import logging
import os
import torch.nn.functional as F
from arch_prior import compute_arch_prior_penalty
from run_manager import *


class ArchSearchConfig:

    def __init__(
        self,
        arch_init_type,
        arch_init_ratio,
        arch_opt_type,
        arch_lr,
        arch_opt_param,
        arch_weight_decay,
        target_hardware,
        ref_value,
    ):
        """architecture parameters initialization & optimizer"""
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            "type": type(self),
        }
        for key in self.__dict__:
            if not key.startswith("_"):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, nBatch):
        raise NotImplementedError

    def build_optimizer(self, params):
        """

        :param params: architecture parameters
        :return: arch_optimizer
        """
        if self.opt_type == "adam":
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError


class GradientArchSearchConfig(ArchSearchConfig):

    def __init__(
        self,
        arch_init_type="normal",
        arch_init_ratio=1e-3,
        arch_opt_type="adam",
        arch_lr=1e-3,
        arch_opt_param=None,
        arch_weight_decay=0,
        target_hardware=None,
        ref_value=None,
        grad_update_arch_param_every=1,
        grad_update_steps=1,
        grad_binary_mode="full",
        grad_data_batch=None,
        grad_reg_loss_type=None,
        grad_reg_loss_params=None,
        grad_meta_enable=False,
        grad_meta_noise_ratio=0.3,
        grad_meta_noise_std=0.15,
        grad_meta_inner_lr=-1.0,
        grad_meta_fd_epsilon=0.01,
        grad_meta_second_order=True,
        grad_meta_real_weight_update=True,
        grad_meta_replay_per_batch=0,
        **kwargs,
    ):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type,
            arch_init_ratio,
            arch_opt_type,
            arch_lr,
            arch_opt_param,
            arch_weight_decay,
            target_hardware,
            ref_value,
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = (
            {} if grad_reg_loss_params is None else grad_reg_loss_params
        )
        # 元学习驱动的架构更新（可选）
        self.meta_enable = bool(grad_meta_enable)
        self.meta_noise_ratio = float(grad_meta_noise_ratio)
        self.meta_noise_std = float(grad_meta_noise_std)
        self.meta_inner_lr = float(grad_meta_inner_lr)
        self.meta_fd_epsilon = float(grad_meta_fd_epsilon)
        self.meta_second_order = bool(grad_meta_second_order)
        self.meta_real_weight_update = bool(grad_meta_real_weight_update)
        self.meta_replay_per_batch = int(grad_meta_replay_per_batch)

        # logging.info(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if nBatch <= 0:
            return schedule
        update_every = max(1, int(self.update_arch_param_every))
        if nBatch < update_every:
            update_every = nBatch
        for i in range(nBatch):
            if (i + 1) % update_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self, ce_loss, expected_value):
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == "mul#log":
            alpha = self.reg_loss_params.get("alpha", 1)
            beta = self.reg_loss_params.get("beta", 0.6)
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == "add#linear":
            reg_lambda = self.reg_loss_params.get("lambda", 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError("Do not support: %s" % self.reg_loss_type)


class RLArchSearchConfig(ArchSearchConfig):

    def __init__(
        self,
        arch_init_type="normal",
        arch_init_ratio=1e-3,
        arch_opt_type="adam",
        arch_lr=1e-3,
        arch_opt_param=None,
        arch_weight_decay=0,
        target_hardware=None,
        ref_value=None,
        rl_batch_size=10,
        rl_update_per_epoch=False,
        rl_update_steps_per_epoch=300,
        rl_baseline_decay_weight=0.99,
        rl_tradeoff_ratio=0.1,
        **kwargs,
    ):
        super(RLArchSearchConfig, self).__init__(
            arch_init_type,
            arch_init_ratio,
            arch_opt_type,
            arch_lr,
            arch_opt_param,
            arch_weight_decay,
            target_hardware,
            ref_value,
        )

        self.batch_size = rl_batch_size
        self.update_per_epoch = rl_update_per_epoch
        self.update_steps_per_epoch = rl_update_steps_per_epoch
        self.baseline_decay_weight = rl_baseline_decay_weight
        self.tradeoff_ratio = rl_tradeoff_ratio

        self._baseline = None
        logging.info(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if nBatch <= 0:
            return schedule
        if self.update_per_epoch:
            schedule[nBatch - 1] = self.update_steps_per_epoch
        else:
            rl_seg_list = get_split_list(nBatch, self.update_steps_per_epoch)
            for j in range(1, len(rl_seg_list)):
                rl_seg_list[j] += rl_seg_list[j - 1]
            for j in rl_seg_list:
                schedule[j - 1] = 1
        return schedule

    def calculate_reward(self, net_info):
        acc = net_info["acc"] / 100
        if self.target_hardware is None:
            return acc
        else:
            # 当测得的硬件指标为 0 或缺失时，回退为仅依赖准确率，避免 0 除错误
            target_val = net_info.get(self.target_hardware, 0)
            if target_val is None or target_val == 0:
                logging.warning(
                    f"calculate_reward: {self.target_hardware} is 0/None, fallback to acc-only reward"
                )
                return acc
            return acc * (
                (self.ref_value / target_val) ** self.tradeoff_ratio
            )

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value


class ArchSearchRunManager:

    def __init__(
        self,
        path,
        super_net,
        run_config: RunConfig,
        arch_search_config: ArchSearchConfig,
        warmup=False,
        task_id=1,
        init_model=True,
        replay_buffer=None,
    ):
        # init weight parameters & build weight_optimizer
        # Super_net is put in RunManager.
        self.run_manager = RunManager(
            path,
            super_net,
            run_config,
            True,
            task_id,
            None,
            init_model,
            replay_buffer=replay_buffer,
            run_phase="nas_search",
        )  # idxs=user_groups[idx]
        self.task_id = task_id
        # use GPU to train net
        self.arch_search_config = arch_search_config

        # init architecture parameters (allow keeping loaded params when continuing)
        if init_model:
            self.net.init_arch_params(
                self.arch_search_config.arch_init_type,
                self.arch_search_config.arch_init_ratio,
            )

        # build architecture optimizer 架构参数
        self.arch_optimizer = self.arch_search_config.build_optimizer(
            self.net.architecture_parameters()
        )

        self.warmup = warmup
        self.warmup_epoch = 0
        self.start_epoch = 0
        self.round = 0
        self.warmup_round = 0
        self.local_train_losses = AverageMeter()
        self.local_valid_losses = AverageMeter()
        self.local_valid_top1 = AverageMeter()
        self.local_train_top1 = AverageMeter()
        self.entropy = AverageMeter()
        self.arch_prior_penalty = AverageMeter()
        self.arch_prior_penalty_last = 0.0

    @property
    def net(self):
        return self.run_manager.net.module

    def write_log(self, log_str, prefix, should_logging_info=True, end="\n"):
        log_fname = os.path.join(
            self.run_manager.logs_path, f"{prefix}_task{self.task_id}.log"
        )
        with open(log_fname, "a") as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_logging_info:
            logging.info(log_str)

    # --------- 公共小工具：蒸馏/正则/正交的准备与复用 --------- #
    def _prepare_distill_model(self, teacher_model):
        if teacher_model is None:
            return None
        model = copy.deepcopy(teacher_model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.to(self.run_manager.device)
        model.eval()
        try:
            self.run_manager.write_log(
                f"[KD] task{self.task_id} teacher prepared: cls={model.__class__.__name__}",
                prefix="train",
                should_print=True,
            )
        except Exception:
            pass
        return model

    def _prepare_reg_anchor(self, teacher_model, reg_lambda, reg_use_ewc):
        anchor_params = None
        fisher = None
        if teacher_model is None or reg_lambda <= 0:
            return anchor_params, fisher
        reg_model = copy.deepcopy(teacher_model)
        if isinstance(reg_model, nn.DataParallel):
            reg_model = reg_model.module
        reg_model = reg_model.to(self.run_manager.device)
        reg_model.eval()
        # 调试：记录 teacher 参数规模，便于确认是否为空
        try:
            total_params = sum(1 for _ in reg_model.named_parameters() if _.requires_grad)
            self.run_manager.write_log(
                f"[REG] task{self.task_id} teacher_cls={reg_model.__class__.__name__}, "
                f"trainable_params={total_params}",
                prefix="train",
                should_print=True,
            )
        except Exception:
            pass
        anchor_params = {}
        for name, p in reg_model.named_parameters():
            if not p.requires_grad:
                continue
            norm_name = name.replace("module.", "", 1)
            anchor_params[norm_name] = p.detach().clone()
        # 若未匹配到任何参数，尝试用 teacher state_dict 与当前模型做交集兜底
        if len(anchor_params) == 0:
            try:
                sd = reg_model.state_dict()
                for name, p in self.run_manager.net.named_parameters():
                    if not p.requires_grad:
                        continue
                    norm_name = name.replace("module.", "", 1)
                    if norm_name in sd and sd[norm_name].shape == p.shape:
                        anchor_params[norm_name] = sd[norm_name].detach().clone()
            except Exception:
                pass
        print(
            f"[REG] task{self.task_id} enable weight anchoring: "
            f"anchor_params={len(anchor_params)}, reg_lambda={reg_lambda}"
        )
        if reg_use_ewc:
            fisher = {}
            sample_loader = list(self.run_manager.run_config.train_loader)
            if len(sample_loader) > 0:
                imgs, labels = sample_loader[0]
                imgs, labels = imgs.to(self.run_manager.device), labels.to(self.run_manager.device)
                reg_model.zero_grad()
                out = reg_model(imgs)
                ce = F.cross_entropy(out, labels)
                ce.backward()
                for name, p in reg_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    norm_name = name.replace("module.", "", 1)
                    if p.grad is None:
                        continue
                    fisher[norm_name] = p.grad.detach().pow(2)
            print(f"[REG] task{self.task_id} EWC fisher entries={len(fisher)}")
        return anchor_params, fisher

    def _compute_kd_ortho_reference_from_loss(self, kd_loss):
        """利用当前 kd_loss 反向得到梯度参考，不改变现有梯度状态。"""
        if kd_loss is None:
            return None
        self.run_manager.optimizer.zero_grad()
        self.run_manager.net.zero_grad()
        kd_loss.backward(retain_graph=True)
        ref = {}
        for name, p in self.run_manager.net.named_parameters():
            if p.grad is not None and p.requires_grad:
                ref[name] = p.grad.detach().clone()
        self.run_manager.optimizer.zero_grad()
        self.run_manager.net.zero_grad()
        return ref

    def _maybe_build_kd_ortho_ref(self, distill_model, eff_kd_lambda, ortho_samples_per_task):
        """预构建 kd_ortho 的全局参考梯度，保持与 RunManager 逻辑一致。"""
        if (
            self.run_manager.cl_ortho_method == "kd_ortho"
            and distill_model is not None
            and eff_kd_lambda > 0
            and ortho_samples_per_task is not None
            and ortho_samples_per_task > 0
        ):
            return self.run_manager.build_kd_ortho_reference(
                max_samples=ortho_samples_per_task
            )
        return None

    def _build_total_loss_for_search(
        self,
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
        """封装 search 阶段的 CE + KD + EWC/锚定，总逻辑与原实现一致。"""
        # 基础 CE
        if self.run_manager.run_config.label_smoothing > 0:
            ce_loss = cross_entropy_with_label_smoothing(
                output, labels, self.run_manager.run_config.label_smoothing
            )
        else:
            ce_loss = self.run_manager.criterion(output, labels)

        # EWC 正则（如果有历史 Fisher）
        ewc_penalty = self.run_manager._ewc_penalty()

        # KD
        kd_loss = None
        if teacher_output is not None and eff_kd_method in ["logit", "logit_conf"] and eff_kd_lambda > 0:
            T = eff_kd_temperature if eff_kd_temperature is not None else 1.0
            student_logp = F.log_softmax(output / T, dim=1)
            teacher_prob = F.softmax(teacher_output / T, dim=1)
            if eff_kd_method == "logit":
                kd_loss = F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (T * T)
            else:
                mask = (teacher_prob > eff_kd_conf).float()
                if mask.sum() > 0:
                    log_teacher = torch.log(teacher_prob + 1e-12)
                    kl_elem = (teacher_prob * (log_teacher - student_logp)) * mask
                    kd_loss = (kl_elem.sum() / mask.sum()) * (T * T)

        # 权重锚定正则（可选 Fisher 权重）
        reg_loss = None
        if reg_anchor_params is not None and reg_lambda > 0:
            reg_loss = 0.0
            skipped = 0
            matched = 0
            for name, p in self.run_manager.net.named_parameters():
                if not p.requires_grad:
                    continue
                norm_name = name.replace("module.", "", 1)
                anchor = reg_anchor_params.get(norm_name)
                if anchor is None or anchor.shape != p.shape:
                    skipped += 1
                    continue
                weight = fisher_params.get(norm_name, 1.0) if fisher_params is not None else 1.0
                if torch.is_tensor(weight):
                    weight = weight.mean().item()
                reg_loss = reg_loss + weight * (p - anchor).pow(2).sum()
                matched += 1
            print(
                f"[REG] task{self.task_id} matched={matched}, skipped={skipped}, "
                f"fisher_used={len(fisher_params) if fisher_params is not None else 0}"
            )
            if matched == 0 and getattr(self, "_reg_warned", False) is False:
                try:
                    self.run_manager.write_log(
                        f"[REG] task{self.task_id} anchor matched 0 (reg_lambda={reg_lambda}); "
                        f"teacher may be None or name mismatch",
                        prefix="train",
                        should_print=True,
                    )
                except Exception:
                    pass
                self._reg_warned = True
                
            # 额外调试：若完全未匹配，打印部分 anchor / 当前参数名以排查名称不一致
            if matched == 0 and not getattr(self, "_reg_debug_logged", False):
                try:
                    anchor_keys = list(reg_anchor_params.keys())[:5] if reg_anchor_params else []
                    cur_keys = []
                    for n, _ in self.run_manager.net.named_parameters():
                        cur_keys.append(n.replace("module.", "", 1))
                        if len(cur_keys) >= 5:
                            break
                    self.run_manager.write_log(
                        f"[REG] task{self.task_id} debug: anchor_keys(sample)={anchor_keys}, "
                        f"cur_param_keys(sample)={cur_keys}",
                        prefix="train",
                        should_print=True,
                    )
                except Exception:
                    pass
                self._reg_debug_logged = True

        total_loss = ce_loss
        if kd_loss is not None and eff_kd_lambda > 0:
            total_loss = total_loss + eff_kd_lambda * kd_loss
        if ewc_penalty is not None and torch.isfinite(ewc_penalty):
            penalty_term = self.run_manager.ewc_lambda * ewc_penalty
            if cl_penalty_clip is not None:
                penalty_term = torch.clamp(penalty_term, max=cl_penalty_clip)
            total_loss = total_loss + penalty_term
        if reg_loss is not None and reg_lambda > 0:
            penalty_term = reg_lambda * reg_loss
            if cl_penalty_clip is not None:
                penalty_term = torch.clamp(penalty_term, max=cl_penalty_clip)
            total_loss = total_loss + penalty_term
        return total_loss, kd_loss, reg_loss, ewc_penalty

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = "%s/checkpoint.pth.tar" % self.run_manager.save_path
            with open(latest_fname, "w") as fout:
                fout.write(model_fname + "\n")

        checkpoint = torch.load(
            model_fname, map_location=torch.device("cpu")
        )  # reduce max gpu-cost

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint["state_dict"])
        self.net.load_state_dict(model_dict)

        if "round" in checkpoint:
            self.round = checkpoint["round"] + 1
        if "warmup" in checkpoint:
            self.warmup = checkpoint["warmup"]
        if self.warmup and "warmup_round" in checkpoint:
            self.warmup_round = checkpoint["warmup_round"] + 1
            
    def load_clients_opt(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, "latest.txt")
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, "r") as fin:
                model_fname = fin.readline()
                if model_fname[-1] == "\n":
                    model_fname = model_fname[:-1]
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = "%s/checkpoint.pth.tar" % self.run_manager.save_path
            with open(latest_fname, "w") as fout:
                fout.write(model_fname + "\n")

        checkpoint = torch.load(
            model_fname, map_location=torch.device("cpu")
        )  # reduce max gpu-cost

        if (
            str(self.run_manager.run_config.data_provider.client_id)
            + "_weight_optimizer"
            in checkpoint
        ):
            self.run_manager.optimizer.load_state_dict(
                checkpoint[
                    str(self.run_manager.run_config.data_provider.client_id)
                    + "_weight_optimizer"
                ]
            )
        if (
            str(self.run_manager.run_config.data_provider.client_id) + "_arch_optimizer"
            in checkpoint
        ):
            self.arch_optimizer.load_state_dict(
                checkpoint[
                    str(self.run_manager.run_config.data_provider.client_id)
                    + "_arch_optimizer"
                ]
            )

    """ training related methods """

    def validate(self, is_test=False, return_top5=True):
        # get performances of current chosen network on validation set
        # self.run_manager.use_gpu()
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = (
            self.run_manager.run_config.test_batch_size
        )
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=is_test, return_top5=return_top5)
        # flops of chosen network
        flops = self.run_manager.net_flops()
        # measure latencies of chosen op

        if self.arch_search_config.target_hardware in [None, "flops"]:
            latency = 0
        if self.arch_search_config.target_hardware == None:
            latency = 0
        else:
            latency, _ = self.run_manager.net_latency(
                l_type=self.arch_search_config.target_hardware, fast=False
            )
        # unused modules back
        self.net.unused_modules_back()
        return valid_res, flops, latency

    def train(
        self,
        fix_net_weights=False,
        server_model=None,
        start_local_epoch=0,
        last_local_epoch=10,
        writer=None,
        preserve_local_arch_params=False,
    ):
        """
        干净的 ProxylessNAS 搜索阶段。
        这里仅保留当前任务上的主任务损失；历史任务回放、EWC、正交更新等
        遗忘缓解逻辑已移出搜索流程，保存在 legacy_forgetting_in_search.py 中。
        """
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                net_dict = self.net.state_dict()
                server_state = server_model.module.state_dict()
                if preserve_local_arch_params:
                    server_state = {
                        key: value
                        for key, value in server_state.items()
                        if "AP_path_alpha" not in key and "AP_path_wb" not in key
                    }
                net_dict.update(server_state)
                self.net.load_state_dict(net_dict)
            else:
                net_dict = self.net.state_dict()
                server_state = server_model.state_dict()
                if preserve_local_arch_params:
                    server_state = {
                        key: value
                        for key, value in server_state.items()
                        if "AP_path_alpha" not in key and "AP_path_wb" not in key
                    }
                net_dict.update(server_state)
                self.net.load_state_dict(net_dict)

        data_loader = self.run_manager.run_config.train_loader
        data_loader = list(data_loader)
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch
        if nBatch == 0:
            client_id = getattr(self.run_manager.run_config.data_provider, "client_id", "unknown")
            self.write_log(
                f"[SearchSkip] task{self.task_id} client{client_id} train loader empty, skip local weight/arch updates",
                prefix="search",
            )

        arch_param_num = len(list(self.net.architecture_parameters()))
        self.entropy = AverageMeter()
        self.arch_prior_penalty = AverageMeter()
        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        if bool(getattr(self, "arch_prior_regularization_enabled", False)) and isinstance(self.arch_search_config, RLArchSearchConfig):
            self.write_log(
                "[ArchPriorLoss] enabled but current arch_algo is RL; prior deviation loss is ignored in RL updates",
                prefix="search",
                should_logging_info=False,
            )
        trn_loss, trn_top1, trn_top5, arch_entropy = 0.0, 0.0, 0.0, 0.0
        val_loss, val_top1, val_top5 = None, None, None
        if self.run_manager.optimizer.param_groups:
            lr = float(self.run_manager.optimizer.param_groups[0].get("lr", 0.0))
        else:
            lr = 0.0
        use_meta_grad = isinstance(self.arch_search_config, GradientArchSearchConfig) and bool(
            getattr(self.arch_search_config, "meta_enable", False)
        )

        for epoch in range(start_local_epoch, last_local_epoch):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            self.run_manager.net.train()
            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update((time.time() - end) / 60)
                
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy（架构探索的熵，用于监控）
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                images_cuda, labels_cuda = None, None
                if not fix_net_weights:
                    images_cuda, labels_cuda = images.cuda(non_blocking=True), labels.cuda(
                        non_blocking=True
                    )
                    if use_meta_grad and epoch > 0 and update_schedule.get(i, 0) > 0:
                        meta_steps = update_schedule.get(i, 0)
                        real_train_loss, real_acc1, real_acc5 = None, None, None
                        for _ in range(meta_steps):
                            _, _, real_train_loss, real_acc1, real_acc5 = self.gradient_step_meta(
                                images_cuda, labels_cuda
                            )
                        if real_train_loss is not None:
                            losses.update(real_train_loss, images_cuda.size(0))
                        if real_acc1 is not None:
                            top1.update(real_acc1, images_cuda.size(0))
                        if real_acc5 is not None:
                            top5.update(real_acc5, images_cuda.size(0))
                    else:
                        self.net.reset_binary_gates()  # random sample binary gates
                        self.net.unused_modules_off()  # remove unused module for speedup
                        output = self.run_manager.net(images_cuda)

                        if self.run_manager.run_config.label_smoothing > 0:
                            total_loss = cross_entropy_with_label_smoothing(
                                output, labels_cuda, self.run_manager.run_config.label_smoothing
                            )
                        else:
                            total_loss = self.run_manager.criterion(output, labels_cuda)

                        acc1, acc5 = accuracy(output, labels_cuda, topk=(1, 5))
                        losses.update(total_loss.item(), images_cuda.size(0))
                        top1.update(acc1[0].item(), images_cuda.size(0))
                        top5.update(acc5[0].item(), images_cuda.size(0))

                        self.run_manager.net.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.run_manager.net.parameters(), max_norm=5.0)
                        self.run_manager.optimizer.step()
                        self.net.unused_modules_back()

                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    if use_meta_grad and not fix_net_weights:
                        # 元学习模式已在上面的 gradient_step_meta 中完成 alpha 更新
                        pass
                    # update architecture parameters according to update_schedule
                    else:
                        for j in range(update_schedule.get(i, 0)):
                            # ---- 强化学习更新 ----
                            if isinstance(self.arch_search_config, RLArchSearchConfig):
                                self.rl_update_step(fast=True)

                            # ---- 梯度更新 ----
                            elif isinstance(
                                self.arch_search_config, GradientArchSearchConfig
                            ):
                                self.gradient_step()

                            else:
                                raise ValueError(
                                    "do not support: %s" % type(self.arch_search_config)
                                )
                self.local_train_losses.update(losses.avg)
                self.local_train_top1.update(top1.avg)
                # measure elapsed time
                batch_time.update((time.time() - end) / 60)
                end = time.time()
            self.entropy.update(entropy.avg)
            trn_loss, trn_top1, trn_top5, arch_entropy = (
                losses.avg,
                top1.avg,
                top5.avg,
                entropy.avg,
            )
            cid = self.run_manager.run_config.data_provider.client_id
            tb_prefix = f"task_{self.task_id}_client_{cid}_"
            writer.add_scalar(tb_prefix + "_entropy", entropy.avg, epoch)
            writer.add_scalar(tb_prefix + "_train_loss", losses.avg, epoch)
            writer.add_scalar(tb_prefix + "_train_top1", top1.avg, epoch)
            writer.add_scalar(tb_prefix + "_train_top5", top5.avg, epoch)
            writer.add_scalar(tb_prefix + "_arch_prior_penalty", self.arch_prior_penalty.avg, epoch)
            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.local_valid_losses.update(val_loss)
                self.local_valid_top1.update(
                    val_top1
                ) 
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                writer.add_scalar(tb_prefix + "_val_loss", val_loss, epoch)
                writer.add_scalar(tb_prefix + "_val_top1", val_top1, epoch)
                writer.add_scalar(tb_prefix + "_val_top5", val_top5, epoch)

        if val_loss is None or val_top1 is None or val_top5 is None:
            # 当 local_epoch_number < validation_frequency 时，这一轮可能从未进入验证分支。
            # 为了保证联邦端聚合日志拿到稳定数值，这里在返回前补做一次验证。
            (val_loss, val_top1, val_top5), flops, latency = self.validate()

        return (
            trn_loss,
            trn_top1,
            trn_top5,
            val_loss,
            val_top1,
            val_top5,
            arch_entropy,
            lr,
        )

    def rl_update_step(self, GlobalCurrentEpoch=1, fast=True):
        assert isinstance(self.arch_search_config, RLArchSearchConfig)
        # prepare data
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = (
            self.run_manager.run_config.test_batch_size
        )
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        # sample nets and get their validation accuracy, latency, etc
        grad_buffer = []
        reward_buffer = []
        net_info_buffer = []
        for i in range(self.arch_search_config.batch_size):
            self.net.reset_binary_gates()  # random sample binary gates
            self.net.unused_modules_off()  # remove unused module for speedup
            # validate the sampled network
            with torch.no_grad():
                output = self.run_manager.net(images)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            net_info = {"acc": acc1[0].item()}
            
            # get additional net info for calculating the reward
            if self.arch_search_config.target_hardware is None:
                pass
            elif self.arch_search_config.target_hardware == "flops":
                net_info["flops"] = self.run_manager.net_flops()
            else:
                net_info[self.arch_search_config.target_hardware], _ = (
                    self.run_manager.net_latency(
                        l_type=self.arch_search_config.target_hardware, fast=fast
                    )
                )
                
            net_info_buffer.append(net_info)
            # calculate reward according to net_info
            reward = self.arch_search_config.calculate_reward(net_info)
            # loss term
            obj_term = 0
            for m in self.net.redundant_modules:
                if m.AP_path_alpha.grad is not None:
                    m.AP_path_alpha.grad.data.zero_()
                obj_term = obj_term + m.log_prob
            loss_term = -obj_term
            # backward
            loss_term.backward()
            # take out gradient dict
            grad_list = []
            for m in self.net.redundant_modules:
                grad_list.append(m.AP_path_alpha.grad.data.clone())
            grad_buffer.append(grad_list)
            reward_buffer.append(reward)
            # unused modules back
            self.net.unused_modules_back()
        # update baseline function
        avg_reward = sum(reward_buffer) / self.arch_search_config.batch_size
        if self.arch_search_config.baseline is None:
            self.arch_search_config.baseline = avg_reward
        else:
            self.arch_search_config.baseline += (
                self.arch_search_config.baseline_decay_weight
                * (avg_reward - self.arch_search_config.baseline)
            )
        # assign gradients
        for idx, m in enumerate(self.net.redundant_modules):
            m.AP_path_alpha.grad.data.zero_()
            for j in range(self.arch_search_config.batch_size):
                m.AP_path_alpha.grad.data += (
                    reward_buffer[j] - self.arch_search_config.baseline
                ) * grad_buffer[j][idx]
            m.AP_path_alpha.grad.data /= self.arch_search_config.batch_size
        # apply gradients
        self.arch_optimizer.step()

    def _compute_train_ce_loss(self, output, labels):
        if self.run_manager.run_config.label_smoothing > 0:
            return cross_entropy_with_label_smoothing(
                output, labels, self.run_manager.run_config.label_smoothing
            )
        return self.run_manager.criterion(output, labels)

    def _compute_expected_hardware_value(self):
        if self.arch_search_config.target_hardware is None:
            return None
        if self.arch_search_config.target_hardware in ["mobile", "cpu", "gpu8"]:
            return self.net.expected_latency()
        if self.arch_search_config.target_hardware == "supernet":
            return None
        if self.arch_search_config.target_hardware == "flops":
            data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            return self.net.expected_flops(input_var)
        raise NotImplementedError

    @staticmethod
    def _add_gaussian_noise(images, noise_ratio: float, noise_std: float):
        if noise_ratio <= 0 or noise_std <= 0:
            return images
        bsz = images.size(0)
        if bsz <= 0:
            return images
        k = int(round(bsz * min(max(noise_ratio, 0.0), 1.0)))
        if k <= 0:
            return images
        noised = images.clone()
        idx = torch.randperm(bsz, device=images.device)[:k]
        noised[idx] = noised[idx] + torch.randn_like(noised[idx]) * noise_std
        return noised

    def _compute_arch_grad_on_train_batch(self, train_images, train_labels, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.run_manager.net.zero_grad()
        self.net.reset_binary_gates()
        self.net.unused_modules_off()
        output = self.run_manager.net(train_images)
        loss = self._compute_train_ce_loss(output, train_labels)
        loss.backward()
        self.net.set_arch_param_grad()
        arch_grads = []
        for p in self.net.architecture_parameters():
            if p.grad is None:
                arch_grads.append(torch.zeros_like(p))
            else:
                arch_grads.append(p.grad.detach().clone())
        self.net.unused_modules_back()
        self.run_manager.net.zero_grad()
        return arch_grads

    def gradient_step_meta(self, train_images, train_labels):
        """
        元学习驱动的架构更新：
        1) 内层：在 meta-train 上做权重虚拟更新得到 W'
        2) 外层：在加噪 meta-val 上计算对 alpha 的元梯度（含有限差分二阶项）
        3) 真实：更新 alpha 后，再在同一 meta-train batch 上真实更新一次 W
        """
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        cfg = self.arch_search_config

        if cfg.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = (
                self.run_manager.run_config.train_batch_size
            )
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = cfg.data_batch
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True

        self.run_manager.net.train()
        MixedEdge.MODE = cfg.binary_mode

        weight_params = [p for p in self.net.weight_parameters() if p.requires_grad]
        arch_params = [p for p in self.net.architecture_parameters() if p.requires_grad]
        if len(weight_params) == 0 or len(arch_params) == 0:
            # 防御分支：参数异常时回退到原始 gradient_step
            base_loss, base_expected = self.gradient_step()
            return base_loss, base_expected, None, None, None

        inner_lr = float(cfg.meta_inner_lr)
        if inner_lr <= 0:
            try:
                inner_lr = float(self.run_manager.optimizer.param_groups[0]["lr"])
            except Exception:
                inner_lr = float(self.run_manager.run_config.init_lr)

        # 可选：拼接回放样本构建 D_pool 的 meta-train batch
        meta_train_images = train_images
        meta_train_labels = train_labels
        replay_k = max(0, int(getattr(cfg, "meta_replay_per_batch", 0)))
        replay_buffer = getattr(self.run_manager, "replay_buffer", None)
        if replay_k > 0 and replay_buffer is not None and len(replay_buffer) > 0:
            rep_x, rep_y = replay_buffer.sample(
                replay_k, mode="global", exclude_task=self.task_id
            )
            if rep_x is not None and rep_y is not None:
                rep_x = rep_x.to(train_images.device, non_blocking=True)
                rep_y = rep_y.to(train_labels.device, non_blocking=True)
                meta_train_images = torch.cat([train_images, rep_x], dim=0)
                meta_train_labels = torch.cat([train_labels, rep_y], dim=0)

        # Step A: 内层虚拟更新 W -> W'
        self.run_manager.net.zero_grad()
        self.net.reset_binary_gates()
        self.net.unused_modules_off()
        inner_output = self.run_manager.net(meta_train_images)
        inner_loss = self._compute_train_ce_loss(inner_output, meta_train_labels)
        inner_grads = torch.autograd.grad(
            inner_loss, weight_params, create_graph=False, allow_unused=True
        )
        self.net.unused_modules_back()
        with torch.no_grad():
            for p, g in zip(weight_params, inner_grads):
                if g is not None:
                    p.add_(-inner_lr * g)

        # Step B: 外层在加噪 meta-val 上计算直接梯度与 v=∇_{W'}L_val
        images_val, labels_val = self.run_manager.run_config.valid_next_batch
        images_val = images_val.cuda(non_blocking=True)
        labels_val = labels_val.cuda(non_blocking=True)
        images_val = self._add_gaussian_noise(
            images_val, cfg.meta_noise_ratio, cfg.meta_noise_std
        )

        self.run_manager.net.zero_grad()
        self.net.reset_binary_gates()
        self.net.unused_modules_off()
        output_val = self.run_manager.net(images_val)
        ce_loss_val = self.run_manager.criterion(output_val, labels_val)
        expected_value = self._compute_expected_hardware_value()
        outer_loss = self.arch_search_config.add_regularization_loss(
            ce_loss_val, expected_value
        )
        prior_penalty_value = 0.0
        if bool(getattr(self, "arch_prior_regularization_enabled", False)):
            target_arch_params = getattr(self, "arch_prior_regularization_target", None)
            lambda_value = float(getattr(self, "arch_prior_regularization_lambda", 0.0))
            if lambda_value > 0 and target_arch_params:
                prior_penalty = compute_arch_prior_penalty(self.net, target_arch_params)
                if prior_penalty is not None:
                    outer_loss = outer_loss + lambda_value * prior_penalty
                    prior_penalty_value = float(prior_penalty.detach().item())
        self.arch_prior_penalty.update(prior_penalty_value, 1)
        self.arch_prior_penalty_last = prior_penalty_value

        outer_loss.backward()
        self.net.set_arch_param_grad()
        direct_arch_grads = []
        for p in arch_params:
            if p.grad is None:
                direct_arch_grads.append(torch.zeros_like(p))
            else:
                direct_arch_grads.append(p.grad.detach().clone())
        val_weight_grads = []
        for p in weight_params:
            if p.grad is None:
                val_weight_grads.append(None)
            else:
                val_weight_grads.append(p.grad.detach().clone())
        self.net.unused_modules_back()
        self.run_manager.net.zero_grad()

        # 恢复原始权重 W（撤销虚拟更新）
        with torch.no_grad():
            for p, g in zip(weight_params, inner_grads):
                if g is not None:
                    p.add_(inner_lr * g)

        # Step C: 有限差分近似二阶项（可开关）
        hessian_arch_grads = [torch.zeros_like(g) for g in direct_arch_grads]
        if bool(cfg.meta_second_order):
            flat_v = [v.reshape(-1) for v in val_weight_grads if v is not None]
            if len(flat_v) > 0:
                v_norm = torch.cat(flat_v).norm().item()
            else:
                v_norm = 0.0
            if v_norm > 0:
                epsilon_base = max(float(cfg.meta_fd_epsilon), 1e-8)
                epsilon = epsilon_base / (v_norm + 1e-12)
                seed = torch.randint(0, 2**31 - 1, (1,), device=train_images.device).item()

                with torch.no_grad():
                    for p, v in zip(weight_params, val_weight_grads):
                        if v is not None:
                            p.add_(epsilon * v)
                grad_alpha_plus = self._compute_arch_grad_on_train_batch(
                    meta_train_images, meta_train_labels, seed=seed
                )

                with torch.no_grad():
                    for p, v in zip(weight_params, val_weight_grads):
                        if v is not None:
                            p.add_(-2.0 * epsilon * v)
                grad_alpha_minus = self._compute_arch_grad_on_train_batch(
                    meta_train_images, meta_train_labels, seed=seed
                )

                with torch.no_grad():
                    for p, v in zip(weight_params, val_weight_grads):
                        if v is not None:
                            p.add_(epsilon * v)

                hessian_arch_grads = [
                    (g_p - g_m) / (2.0 * epsilon)
                    for g_p, g_m in zip(grad_alpha_plus, grad_alpha_minus)
                ]

        # Step D: 组装元梯度并更新 alpha
        self.run_manager.net.zero_grad()
        for p, g_direct, g_hessian in zip(
            arch_params, direct_arch_grads, hessian_arch_grads
        ):
            p.grad = g_direct - inner_lr * g_hessian
        self.arch_optimizer.step()
        if MixedEdge.MODE == "two":
            self.net.rescale_updated_arch_param()

        # Step E: 在同一 meta-train batch 上真实更新一次 W
        real_train_loss = None
        real_acc1 = None
        real_acc5 = None
        if bool(cfg.meta_real_weight_update):
            self.run_manager.net.zero_grad()
            self.net.reset_binary_gates()
            self.net.unused_modules_off()
            output_train_real = self.run_manager.net(meta_train_images)
            total_loss = self._compute_train_ce_loss(output_train_real, meta_train_labels)
            acc1, acc5 = accuracy(output_train_real, meta_train_labels, topk=(1, 5))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.run_manager.net.parameters(), max_norm=5.0)
            self.run_manager.optimizer.step()
            self.net.unused_modules_back()
            real_train_loss = float(total_loss.detach().item())
            real_acc1 = float(acc1[0].item())
            real_acc5 = float(acc5[0].item())

        MixedEdge.MODE = None
        return (
            float(outer_loss.detach().item()),
            expected_value.item() if expected_value is not None else None,
            real_train_loss,
            real_acc1,
            real_acc5,
        )

    def gradient_step(
        self,
    ):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        if self.arch_search_config.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = (
                self.run_manager.run_config.train_batch_size
            )
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = (
                self.arch_search_config.data_batch
            )
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        time2 = time.time()  # time
        # compute output
        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup
        output = self.run_manager.net(images)
        time3 = time.time()  # time
        # loss
        ce_loss = self.run_manager.criterion(output, labels)
        expected_value = self._compute_expected_hardware_value()
        loss = self.arch_search_config.add_regularization_loss(ce_loss, expected_value)
        prior_penalty_value = 0.0
        if bool(getattr(self, "arch_prior_regularization_enabled", False)):
            target_arch_params = getattr(self, "arch_prior_regularization_target", None)
            lambda_value = float(getattr(self, "arch_prior_regularization_lambda", 0.0))
            if lambda_value > 0 and target_arch_params:
                prior_penalty = compute_arch_prior_penalty(self.net, target_arch_params)
                if prior_penalty is not None:
                    loss = loss + lambda_value * prior_penalty
                    prior_penalty_value = float(prior_penalty.detach().item())
        self.arch_prior_penalty.update(prior_penalty_value, 1)
        self.arch_prior_penalty_last = prior_penalty_value
        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        self.net.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == "two":
            self.net.rescale_updated_arch_param()
        # back to normal mode
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        return loss.data.item(), (
            expected_value.item() if expected_value is not None else None
        )

    def warm_up(
        self,
        warmup_epochs=25,
        server_model=None,
        start_local_epoch=None,
        last_local_epoch=None,
        writer=None,
        preserve_local_arch_params=False,
    ):
        # ========== Warmup 总览：用当前训练集预训练权重，不更新架构参数 ==========
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                net_dict = self.net.state_dict()
                server_state = server_model.module.state_dict()
                if preserve_local_arch_params:
                    server_state = {
                        key: value
                        for key, value in server_state.items()
                        if "AP_path_alpha" not in key and "AP_path_wb" not in key
                    }
                net_dict.update(server_state)
                self.net.load_state_dict(net_dict)
            else:
                net_dict = self.net.state_dict()
                server_state = server_model.state_dict()
                if preserve_local_arch_params:
                    server_state = {
                        key: value
                        for key, value in server_state.items()
                        if "AP_path_alpha" not in key and "AP_path_wb" not in key
                    }
                net_dict.update(server_state)
                self.net.load_state_dict(net_dict)
        
        # 预准备：不做权重锚定/EWC/正则/正交/蒸馏
        # distill_model = None
        # if teacher_model is not None:
        #     distill_model = copy.deepcopy(teacher_model)
        #     if isinstance(distill_model, nn.DataParallel):
        #         distill_model = distill_model.module
        #     distill_model = distill_model.to(self.run_manager.device)
        #     distill_model.eval()
            
        # eff_kd_lambda = cl_kd_lambda if cl_kd_lambda is not None else kd_lambda
        # eff_kd_temperature = cl_kd_temperature if cl_kd_temperature is not None else kd_temperature
        # eff_kd_method = (cl_kd_method or "none").lower()
        # eff_kd_conf = cl_kd_conf_threshold if cl_kd_conf_threshold is not None else 0.5
        # reg_lambda = reg_lambda if reg_lambda > 0 else getattr(self.run_manager.run_config, "ewc_lambda", 0.0)
        # reg_anchor_params, fisher_params = None, None
        # kd_ortho_ref_grads = None
    
        # 日志文件：记录每个 batch 被激活的候选算子
        ops_log_path = os.path.join(
            self.run_manager.save_path, "logs", "warmup_active_ops.log"
        )
        os.makedirs(os.path.dirname(ops_log_path), exist_ok=True)
        ops_log_f = open(ops_log_path, "a", buffering=1)
        ops_log_f.write("# warmup active ops log\n")

        # 指标记录
        val_loss, val_top1, val_top5, warmup_lr = None, None, None, None
        losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
        
        # 数据贮备
        lr_max = 0.025
        data_loader = self.run_manager.run_config.train_loader
        data_loader = list(data_loader)
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        # ---- 每个 epoch 的权重训练 ----
        for epoch in range(start_local_epoch, last_local_epoch):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
            self.run_manager.net.train()
            end = time.time()

            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                
                # ---- lr动态调整 ---- 
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                images, labels = images.cuda(non_blocking=True), labels.cuda(
                    non_blocking=True
                )
                
                # ---- 前向：随机采样子网并 forward ---- 
                self.net.reset_binary_gates()  # 随机选 path
                self.net.unused_modules_off()  # 移除未选 path 加速

                # 记录本 batch 激活的候选算子，便于后续分析抽样分布
                if ops_log_f is not None:
                    active_ops = []
                    for m in self.net.redundant_modules:
                        try:
                            idx = m.active_index[0]
                            if isinstance(idx, tuple):
                                idx = idx[0]
                            op = m.candidate_ops[idx]
                            active_ops.append(op.module_str)
                        except Exception as e:
                            active_ops.append(f"ERR:{type(m).__name__}")
                    ops_log_f.write(
                        f"epoch={epoch} batch={i} ops=[{'; '.join(active_ops)}]\n"
                    )

                output = self.run_manager.net(images)
                
                # 检测非有限输出
                if not torch.isfinite(output).all():
                    print(f"[Warmup] non-finite output detected at epoch {epoch} batch {i}, clamp to finite")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                    
                # ---- ce损失计算 ---- 
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)
                
                # 检测非有限损失
                total_loss = ce_loss
                if not torch.isfinite(total_loss):
                    # 本轮总 loss 不可用，跳过但释放关闭的模块
                    print(f"[Warmup] non-finite total_loss at epoch {epoch} batch {i}, skip step")
                    self.net.unused_modules_back()
                    continue
                
                # ---- 统计指标 ----
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(total_loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                
                # ---- 反向传播 & 权重更新 ----
                self.run_manager.net.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.run_manager.net.parameters(), max_norm=5.0)
                
                # 更新选中path的权重
                self.run_manager.optimizer.step()  
                # 恢复未选中path
                self.net.unused_modules_back()
            
                # 时间统计
                batch_time.update(time.time() - end)
                end = time.time()
                
            # ---- epoch 结束，记录指标 & 验证 ----
            (val_loss, val_top1, val_top5), flops, latency = self.validate()
            cid = self.run_manager.run_config.data_provider.client_id
            tb_prefix = f"task_{self.task_id}_client_{cid}_"
            writer.add_scalar(tb_prefix + "_warmup_train_loss", losses.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_train_top1", top1.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_train_top5", top5.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_loss", val_loss, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_top1", val_top1, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_top5", val_top5, epoch)
        if ops_log_f is not None:
            ops_log_f.close()
        return losses.avg, top1.avg, top5.avg, val_loss, val_top1, val_top5, warmup_lr

    def ReturnServerModelWeight(self):
        return self.net.state_dict()

    def get_model(self):
        return self.net

    def get_local_data_weight(self):
        return self.run_manager.run_config.data_provider.trn_set_length

    def inference(self, server_model=None, is_test=False, return_top5=True):

        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.load_state_dict(server_model.module.state_dict())
            else:
                self.net.load_state_dict(server_model.state_dict())
        if return_top5:
            is_test_v = is_test
            (val_loss, val_top1, val_top5), flops, latency = self.validate(
                is_test=is_test_v, return_top5=return_top5
            )
            return val_loss, val_top1, val_top5
        else:
            (val_loss, val_top1), flops, latency = self.validate(
                is_test=is_test, return_top5=return_top5
            )
            return val_loss, val_top1

    def LocalTrainTop1(self):
        return self.local_train_top1.avg

    def LocalTrainLosses(self):
        return self.local_train_losses.avg

    def LocalValidTop1(self):
        return self.local_valid_top1.avg

    def LocalValidLosses(self):
        return self.local_valid_losses.avg

    def get_normal_net(self):
        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        logging.info(
            "Total training params: %.2fM" % (count_parameters(normal_net) / 1e6)
        )
        os.makedirs(os.path.join(self.run_manager.path, "learned_net"), exist_ok=True)
        json.dump(
            normal_net.config,
            open(os.path.join(self.run_manager.path, "learned_net/net.config"), "w"),
            indent=4,
            default=set_to_list,
        )
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, "learned_net/run.config"), "w"),
            indent=4,
            default=set_to_list,
        )
        torch.save(
            {
                "state_dict": normal_net.state_dict(),
                "dataset": self.run_manager.run_config.dataset,
            },
            os.path.join(self.run_manager.path, "learned_net/init"),
        )
        # ----new add----
        self.run_manager.save_model(
            {
                "state_dict": normal_net.state_dict(),  # 并行后的修改
            },
            HardwareClass="PlNAS",
        )


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
