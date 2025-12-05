# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import logging
import torch.nn.functional as F
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

        # logging.info(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if nBatch < self.update_arch_param_every:
            self.update_arch_param_every = nBatch
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
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
    ):
        # init weight parameters & build weight_optimizer
        # Super_net is put in RunManager.
        self.run_manager = RunManager(
            path, super_net, run_config, True, task_id, None, init_model
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
        teacher_model=None,
        kd_lambda=0.0,
        kd_temperature=1.0,
        reg_lambda=0.0,
        reg_use_ewc=False,
        cl_kd_method="none",
        cl_kd_lambda=None,
        cl_kd_temperature=None,
        cl_kd_conf_threshold=0.5,
        cl_ortho_method="none",
        cl_ortho_scale=1.0,
        ortho_samples_per_task=0,
        cl_penalty_clip=None,
    ):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                net_dict = self.net.state_dict()
                net_dict.update(server_model.module.state_dict())
                self.net.load_state_dict(net_dict)
            else:
                net_dict = self.net.state_dict()
                net_dict.update(server_model.state_dict())
                self.net.load_state_dict(net_dict)

        # teacher_model 用于跨任务蒸馏/锚定
        distill_model = None
        if teacher_model is not None:
            distill_model = copy.deepcopy(teacher_model)
            if isinstance(distill_model, nn.DataParallel):
                distill_model = distill_model.module
            distill_model = distill_model.to(self.run_manager.device)
            distill_model.eval()
        # baseline 风格的 KD 配置：优先 cl_kd_*，回退旧 kd_*
        eff_kd_lambda = cl_kd_lambda if cl_kd_lambda is not None else kd_lambda
        eff_kd_temperature = cl_kd_temperature if cl_kd_temperature is not None else kd_temperature
        eff_kd_method = (cl_kd_method or "none").lower()
        eff_kd_conf = cl_kd_conf_threshold if cl_kd_conf_threshold is not None else 0.5
        if distill_model is not None and eff_kd_lambda > 0 and eff_kd_method != "none":
            print(
                f"[KD] task{self.task_id} enable distillation: "
                f"method={eff_kd_method}, lambda={eff_kd_lambda}, T={eff_kd_temperature}, conf={eff_kd_conf}"
            )
        # reg_anchor_params/fisher_params 用于“权重锚定 + 可选 EWC”正则，避免跨任务漂移
        reg_anchor_params = None
        fisher_params = None

        if teacher_model is not None and reg_lambda > 0:
            reg_model = copy.deepcopy(teacher_model)
            if isinstance(reg_model, nn.DataParallel):
                reg_model = reg_model.module
            reg_model = reg_model.to(self.run_manager.device)
            reg_model.eval()
            # 仅锚定可训练参数（不含 buffer），避免 shape 不一致
            reg_anchor_params = {}
            for name, p in reg_model.named_parameters():
                if not p.requires_grad:
                    continue
                norm_name = name.replace("module.", "", 1)
                reg_anchor_params[norm_name] = p.detach().clone()
            print(
                f"[REG] task{self.task_id} enable weight anchoring: "
                f"anchor_params={len(reg_anchor_params)}, reg_lambda={reg_lambda}"
            )
            
            if reg_use_ewc:
                # 可选：用 teacher 的首个 batch 近似 Fisher，对重要参数给予更大权重
                fisher_params = {}
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
                        fisher_params[norm_name] = p.grad.detach().pow(2)
                print(
                    f"[REG] task{self.task_id} EWC fisher entries={len(fisher_params)}"
                )

        data_loader = self.run_manager.run_config.train_loader
        data_loader = list(data_loader)
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters()))
        self.entropy = AverageMeter()
        update_schedule = self.arch_search_config.get_update_schedule(nBatch)
        trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, arch_entropy = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        # kd_ortho 全局参考梯度，与 RunManager 逻辑保持一致
        kd_ortho_ref_grads = None
        if (
            self.run_manager.cl_ortho_method == "kd_ortho"
            and distill_model is not None
            and eff_kd_lambda > 0
            and ortho_samples_per_task > 0
        ):
            kd_ortho_ref_grads = self.run_manager.build_kd_ortho_reference(
                max_samples=ortho_samples_per_task
            )
        # 小工具：计算本 batch 的各项损失，保持与 RunManager 中训练逻辑一致
        def _build_total_loss(output, labels, teacher_output):
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
        
        for epoch in range(start_local_epoch, last_local_epoch):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # 切换到训练模式；weight/arch 交替更新
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update((time.time() - end) / 60)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy（架构探索的熵，用于监控）
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                if not fix_net_weights:
                    images, labels = images.cuda(non_blocking=True), labels.cuda(
                        non_blocking=True
                    )
                    # 前向：随机 sample 二进制门，关闭未用模块加速
                    self.net.reset_binary_gates()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup
                    output = self.run_manager.net(images)
                    
                    # teacher 输出（若设置蒸馏）
                    teacher_output = None
                    if distill_model is not None and eff_kd_lambda > 0 and eff_kd_method != "none":
                        with torch.no_grad():
                            teacher_output = distill_model(images)
                    # loss: CE + 可选 KD + 可选锚定正则
                    total_loss, kd_loss, reg_loss, ewc_penalty = _build_total_loss(output, labels, teacher_output)
                    # 正交更新（仅支持 kd_ortho，与 baseline 对齐）
                    old_grads = None
                    if (
                        cl_ortho_method == "kd_ortho"
                        and teacher_output is not None
                        and kd_loss is not None
                        and eff_kd_lambda > 0
                    ):
                        if kd_ortho_ref_grads is not None:
                            old_grads = {n: g.to(self.run_manager.device) for n, g in kd_ortho_ref_grads.items()}
                        else:
                            self.run_manager.optimizer.zero_grad()
                            self.run_manager.net.zero_grad()
                            kd_loss.backward(retain_graph=True)
                            old_grads = {}
                            for name, p in self.run_manager.net.named_parameters():
                                if p.grad is not None and p.requires_grad:
                                    old_grads[name] = p.grad.detach().clone()
                            self.run_manager.optimizer.zero_grad()
                            self.run_manager.net.zero_grad()
                        
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(total_loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    # compute gradient and do SGD step
                    self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                    total_loss.backward()
                    # 若需要，投影梯度到 teacher 引导的正交子空间
                    if old_grads is not None:
                        eps = 1e-12
                        for name, p in self.run_manager.net.named_parameters():
                            if p.grad is None or name not in old_grads:
                                continue
                            g = p.grad
                            g_old = old_grads[name].to(g.device)
                            denom = torch.dot(g_old.view(-1), g_old.view(-1)) + eps
                            if denom.item() == 0.0:
                                continue
                            proj = torch.dot(g.view(-1), g_old.view(-1)) / denom
                            g_ortho = g.view(-1) - cl_ortho_scale * proj * g_old.view(-1)
                            p.grad.copy_(g_ortho.view_as(p))
                    torch.nn.utils.clip_grad_norm_(self.run_manager.net.parameters(), max_norm=5.0)
                    self.run_manager.optimizer.step()  # update weight parameters
                    # unused modules back
                    self.net.unused_modules_back()
                    
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        if isinstance(self.arch_search_config, RLArchSearchConfig):
                            # [1]. Reinforcement Update
                            self.rl_update_step(fast=True)

                        elif isinstance(
                            self.arch_search_config, GradientArchSearchConfig
                        ):
                            # [2]. Gradient Update
                            self.gradient_step()
                            # [!] Update arch_optimizer

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
            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.local_valid_losses.update(val_loss)
                self.local_valid_top1.update(
                    val_top1
                )  # 需要.avg吗？数据结构为什么？top1才要.avg
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                writer.add_scalar(tb_prefix + "_val_loss", val_loss, epoch)
                writer.add_scalar(tb_prefix + "_val_top1", val_top1, epoch)
                writer.add_scalar(tb_prefix + "_val_top5", val_top5, epoch)
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
        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == "mobile":
            expected_value = self.net.expected_latency()
        elif self.arch_search_config.target_hardware == "cpu":
            expected_value = self.net.expected_latency()
            # logging.info('cpu latency:', expected_value)
        elif self.arch_search_config.target_hardware == "gpu8":
            expected_value = self.net.expected_latency()
        elif self.arch_search_config.target_hardware == "supernet":
            expected_value = None
        elif self.arch_search_config.target_hardware == "flops":
            data_shape = [1] + list(
                self.run_manager.run_config.data_provider.data_shape
            )
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.expected_flops(input_var)
        else:
            raise NotImplementedError
        loss = self.arch_search_config.add_regularization_loss(ce_loss, expected_value)
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
        teacher_model=None,
        kd_lambda=0.0,
        kd_temperature=1.0,
        reg_lambda=0.0,
        reg_use_ewc=False,
        cl_kd_method="none",
        cl_kd_lambda=None,
        cl_kd_temperature=None,
        cl_kd_conf_threshold=0.5,
        cl_ortho_method="none",
        cl_ortho_scale=1.0,
        ortho_samples_per_task=0,
        cl_penalty_clip=None,
    ):
        if server_model != None:
            if isinstance(server_model, nn.DataParallel):
                self.net.load_state_dict(server_model.module.state_dict())
            else:
                self.net.load_state_dict(server_model.state_dict())
        # teacher for warmup KD/REG
        distill_model = None
        if teacher_model is not None:
            distill_model = copy.deepcopy(teacher_model)
            if isinstance(distill_model, nn.DataParallel):
                distill_model = distill_model.module
            distill_model = distill_model.to(self.run_manager.device)
            distill_model.eval()
        eff_kd_lambda = cl_kd_lambda if cl_kd_lambda is not None else kd_lambda
        eff_kd_temperature = cl_kd_temperature if cl_kd_temperature is not None else kd_temperature
        eff_kd_method = (cl_kd_method or "none").lower()
        eff_kd_conf = cl_kd_conf_threshold if cl_kd_conf_threshold is not None else 0.5
        reg_lambda = reg_lambda if reg_lambda > 0 else getattr(self.run_manager.run_config, "ewc_lambda", 0.0)
        reg_anchor_params, fisher_params = None, None
        if distill_model is not None and reg_lambda > 0:
            reg_model = copy.deepcopy(distill_model)
            if isinstance(reg_model, nn.DataParallel):
                reg_model = reg_model.module
            reg_model = reg_model.to(self.run_manager.device)
            reg_model.eval()
            reg_anchor_params = {}
            for name, p in reg_model.named_parameters():
                if p.requires_grad:
                    reg_anchor_params[name] = p.detach().clone()
            if reg_use_ewc:
                fisher_params = {}
                sample_loader = list(self.run_manager.run_config.train_loader)
                if len(sample_loader) > 0:
                    imgs, labels = sample_loader[0]
                    imgs, labels = imgs.to(self.run_manager.device), labels.to(self.run_manager.device)
                    reg_model.zero_grad()
                    out = reg_model(imgs)
                    ce = F.cross_entropy(out, labels)
                    ce.backward()
                    for name, p in reg_model.named_parameters():
                        if p.grad is not None and p.requires_grad:
                            fisher_params[name] = p.grad.detach().pow(2)
        lr_max = 0.025
        val_loss, val_top1, val_top5, warmup_lr = None, None, None, None
        losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
        data_loader = self.run_manager.run_config.train_loader
        data_loader = list(data_loader)
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        # 逐epoch进行训练
        kd_ortho_ref_grads = None
        if (
            cl_ortho_method == "kd_ortho"
            and distill_model is not None
            and eff_kd_lambda > 0
            and ortho_samples_per_task > 0
        ):
            kd_ortho_ref_grads = self.run_manager.build_kd_ortho_reference(
                max_samples=ortho_samples_per_task
            )
        for epoch in range(start_local_epoch, last_local_epoch):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()

            # epoch分解出batch中的训练
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr动态调整
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                images, labels = images.cuda(non_blocking=True), labels.cuda(
                    non_blocking=True
                )
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates  随机选path  from supernet
                self.net.unused_modules_off()  # remove unused module for speedup  移除未选的path
                output = self.run_manager.net(images)  # forward (DataParallel)
                if not torch.isfinite(output).all():
                    print(f"[Warmup] non-finite output detected at epoch {epoch} batch {i}, clamp to finite")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    ce_loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    ce_loss = self.run_manager.criterion(output, labels)
                if not torch.isfinite(ce_loss):
                    print(f"[Warmup] non-finite ce_loss detected at epoch {epoch} batch {i}, skip step")
                    self.net.unused_modules_back()
                    continue
                ewc_penalty = self.run_manager._ewc_penalty()
                kd_loss = None
                teacher_output = None
                if distill_model is not None and eff_kd_lambda > 0 and eff_kd_method in ["logit", "logit_conf"]:
                    with torch.no_grad():
                        teacher_output = distill_model(images)
                        if not torch.isfinite(teacher_output).all():
                            teacher_output = torch.nan_to_num(teacher_output, nan=0.0, posinf=1e4, neginf=-1e4)
                    T = eff_kd_temperature if eff_kd_temperature is not None else 1.0
                    s_logp = F.log_softmax(output / T, dim=1)
                    t_prob = F.softmax(teacher_output / T, dim=1)
                    if eff_kd_method == "logit":
                        kd_loss = F.kl_div(s_logp, t_prob, reduction="batchmean") * (T * T)
                    else:
                        mask = (t_prob > eff_kd_conf).float()
                        if mask.sum() > 0:
                            log_t = torch.log(t_prob + 1e-12)
                            kd_loss = ((t_prob * (log_t - s_logp)) * mask).sum() / mask.sum()
                            kd_loss = kd_loss * (T * T)
                    if kd_loss is not None and not torch.isfinite(kd_loss):
                        print(f"[Warmup] non-finite kd_loss at epoch {epoch} batch {i}, drop KD term")
                        kd_loss = None
                reg_loss = None
                if reg_anchor_params is not None and reg_lambda > 0:
                    reg_loss = 0.0
                    for name, p in self.run_manager.net.named_parameters():
                        if not p.requires_grad:
                            continue
                        anchor = reg_anchor_params.get(name.replace("module.", "", 1))
                        if anchor is None or anchor.shape != p.shape:
                            continue
                        weight = fisher_params.get(name.replace("module.", "", 1), 1.0) if fisher_params is not None else 1.0
                        if torch.is_tensor(weight):
                            weight = weight.mean().item()
                        reg_loss = reg_loss + weight * (p - anchor).pow(2).sum()
                    reg_loss_tensor = reg_loss if torch.is_tensor(reg_loss) else torch.tensor(reg_loss, device=self.run_manager.device)
                    if not torch.isfinite(reg_loss_tensor):
                        print(f"[Warmup] non-finite reg_loss at epoch {epoch} batch {i}, drop REG term")
                        reg_loss = None
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
                if not torch.isfinite(total_loss):
                    # 本轮总 loss 不可用，跳过但释放关闭的模块
                    print(f"[Warmup] non-finite total_loss at epoch {epoch} batch {i}, skip step")
                    self.net.unused_modules_back()
                    continue
                old_grads = None
                if (
                    cl_ortho_method == "kd_ortho"
                    and kd_loss is not None
                    and eff_kd_lambda > 0
                    and distill_model is not None
                ):
                    if kd_ortho_ref_grads is not None:
                        old_grads = {n: g.to(self.run_manager.device) for n, g in kd_ortho_ref_grads.items()}
                    else:
                        self.run_manager.optimizer.zero_grad()
                        self.run_manager.net.zero_grad()
                        kd_loss.backward(retain_graph=True)
                        old_grads = {}
                        for name, p in self.run_manager.net.named_parameters():
                            if p.grad is not None and p.requires_grad:
                                old_grads[name] = p.grad.detach().clone()
                        self.run_manager.optimizer.zero_grad()
                        self.run_manager.net.zero_grad()
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(total_loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                total_loss.backward()
                if old_grads is not None:
                    eps = 1e-12
                    for name, p in self.run_manager.net.named_parameters():
                        if p.grad is None or name not in old_grads:
                            continue
                        g = p.grad
                        g_old = old_grads[name].to(g.device)
                        denom = torch.dot(g_old.view(-1), g_old.view(-1)) + eps
                        if denom.item() == 0.0:
                            continue
                        proj = torch.dot(g.view(-1), g_old.view(-1)) / denom
                        g_ortho = g.view(-1) - cl_ortho_scale * proj * g_old.view(-1)
                        p.grad.copy_(g_ortho.view_as(p))
                torch.nn.utils.clip_grad_norm_(self.run_manager.net.parameters(), max_norm=5.0)
                self.run_manager.optimizer.step()  # update weight parameters  更新选中path的权重
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            (val_loss, val_top1, val_top5), flops, latency = self.validate()
            cid = self.run_manager.run_config.data_provider.client_id
            tb_prefix = f"task_{self.task_id}_client_{cid}_"
            writer.add_scalar(tb_prefix + "_warmup_train_loss", losses.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_train_top1", top1.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_train_top5", top5.avg, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_loss", val_loss, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_top1", val_top1, epoch)
            writer.add_scalar(tb_prefix + "_warmup_val_top5", val_top5, epoch)
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
