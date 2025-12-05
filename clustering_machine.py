import pickle
import time
import warnings

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


class ClusteringMachine:
    def __init__(self, target_hardware=None, config=None, global_server=None,
                 clients_idx_arr=None, clients=None, start_round=0,
                 last_round=None, path='./', task_id=1, teacher_model=None):
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
        self.teacher_model = teacher_model
        self._logs_path, self._save_path = None, None
        
        # TensorBoard 日志路径中加入 task_id
        if self.hardware is not None:
            self.writerTf = SummaryWriter(logdir=os.path.join(self.path, 'tensorboard'), 
                                          comment=f"{self.hardware}_fed_search_task_{task_id}")
        else:
            self.writerTf = SummaryWriter(logdir=os.path.join(self.path, 'tensorboard'), 
                                          comment=f"fed_search_task_{task_id}")
        print('tensorboardX logdir', self.writerTf.logdir)

    
    def train_clients(self):
        self.start_round = self.global_server.round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        best_val_acc = 0
        kd_lambda = getattr(self.config, "kd_lambda", 0.0)
        kd_temperature = getattr(self.config, "kd_temperature", 1.0)
        reg_lambda = getattr(self.config, "reg_lambda", 0.0)
        reg_use_ewc = getattr(self.config, "reg_use_ewc", False)
        # baseline 风格的 CL 参数（若未显式配置，保留旧 supernet 超参）
        cl_kd_method = getattr(self.config, "cl_kd_method", "none")
        cl_kd_lambda = getattr(self.config, "cl_kd_logit_lambda", 0.0) or kd_lambda
        cl_kd_temperature = getattr(self.config, "cl_kd_temperature", kd_temperature)
        cl_kd_conf_threshold = getattr(self.config, "cl_kd_conf_threshold", 0.5)
        cl_ortho_method = getattr(self.config, "cl_ortho_method", "none")
        cl_ortho_scale = getattr(self.config, "cl_ortho_scale", 1.0)
        ortho_samples_per_task = getattr(self.config, "ortho_samples_per_task", 0)
        cl_penalty_clip = getattr(self.config, "cl_penalty_clip", None)
        # ewc_lambda 可作为 reg_lambda 的替代开关，方便与 baseline 对齐
        reg_lambda = reg_lambda if reg_lambda > 0 else getattr(self.config, "ewc_lambda", 0.0)
        for round in range(self.start_round, self.last_round):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_entropy, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            # 拿当前全局超网权重，作为本轮下发给各客户端的初始模型
            server_model = copy.deepcopy(self.global_server.net)
            teacher_for_clients = self.teacher_model if self.teacher_model is not None else server_model
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, entropy, lr = self.clients[
                    idx].train(
                    server_model=server_model,
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    writer=self.writerTf,
                    teacher_model=teacher_for_clients,
                    kd_lambda=kd_lambda,
                    kd_temperature=kd_temperature,
                    reg_lambda=reg_lambda,
                    reg_use_ewc=reg_use_ewc,
                    cl_kd_method=cl_kd_method,
                    cl_kd_lambda=cl_kd_lambda,
                    cl_kd_temperature=cl_kd_temperature,
                    cl_kd_conf_threshold=cl_kd_conf_threshold,
                    cl_ortho_method=cl_ortho_method,
                    cl_ortho_scale=cl_ortho_scale,
                    ortho_samples_per_task=ortho_samples_per_task,
                    cl_penalty_clip=cl_penalty_clip)
                # local_weight 表示该客户端本轮可用的训练样本数，聚合时作为加权系数
                local_weight = self.clients[idx].get_local_data_weight()
                if local_weight <= 0:
                    # 跳过无数据客户端，避免聚合时 total_weight=0 -> NaN
                    self.write_log(f"skip client {idx} in round {round} because local data weight is 0", prefix='search')
                    continue
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(local_weight)
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_entropy.update(entropy)
                clients_lr.update(lr)
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
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.run_manager.net.module.load_state_dict(server_dict)
            
            self.global_server.write_log('-' * 30 + 'Current Architecture [%d]' % (round + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.global_server.net.blocks):
                self.global_server.write_log('%d. %s' % (idx, block.module_str), prefix='arch')

            # Calculate avg training accuracy over all users at every round
            self.global_server.net.eval()
            with torch.no_grad():
                # directly use global_server's centralized data for faster inference
                val_loss, val_acc_top1, val_acc_top5 = self.global_server.inference(is_test=False)
                if best_val_acc < val_acc_top1:
                    best_val_acc = val_acc_top1
                    is_best = True
                else:
                    is_best = False
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
            self.global_server.run_manager.save_model(checkpoint, is_best=is_best, model_name="global.pth.tar")
                
            # self.test_inference()  # 测试集上跑一下
        self.writerTf.close()

    
    def warmup_clients(self):
        self.warmup_round = self.global_server.warmup_round
        print('len(self.clients_idx_arr): ', len(self.clients_idx_arr))
        kd_lambda = getattr(self.config, "kd_lambda", 0.0)
        kd_temperature = getattr(self.config, "kd_temperature", 1.0)
        reg_lambda = getattr(self.config, "reg_lambda", 0.0)
        reg_use_ewc = getattr(self.config, "reg_use_ewc", False)
        cl_kd_method = getattr(self.config, "cl_kd_method", "none")
        cl_kd_lambda = getattr(self.config, "cl_kd_logit_lambda", 0.0) or kd_lambda
        cl_kd_temperature = getattr(self.config, "cl_kd_temperature", kd_temperature)
        cl_kd_conf_threshold = getattr(self.config, "cl_kd_conf_threshold", 0.5)
        cl_ortho_method = getattr(self.config, "cl_ortho_method", "none")
        cl_ortho_scale = getattr(self.config, "cl_ortho_scale", 1.0)
        ortho_samples_per_task = getattr(self.config, "ortho_samples_per_task", 0)
        cl_penalty_clip = getattr(self.config, "cl_penalty_clip", None)
        reg_lambda = reg_lambda if reg_lambda > 0 else getattr(self.config, "ewc_lambda", 0.0)

        for round in range(self.warmup_round, self.config.warmup_n_rounds):
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            clients_params_arr, clients_data_w = [], []
            server_model = copy.deepcopy(self.global_server.net)
            teacher_for_clients = self.teacher_model if self.teacher_model is not None else server_model
            round_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                # 预热阶段客户端的训练：随机子网前后向，只更新权重，不更新架构参数
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[
                    idx].warm_up(server_model=server_model, start_local_epoch=start_local_epoch,
                                 last_local_epoch=last_local_epoch, writer=self.writerTf,
                                 teacher_model=teacher_for_clients,
                                 kd_lambda=kd_lambda,
                                 kd_temperature=kd_temperature,
                                 reg_lambda=reg_lambda,
                                 reg_use_ewc=reg_use_ewc,
                                 cl_kd_method=cl_kd_method,
                                 cl_kd_lambda=cl_kd_lambda,
                                 cl_kd_temperature=cl_kd_temperature,
                                 cl_kd_conf_threshold=cl_kd_conf_threshold,
                                 cl_ortho_method=cl_ortho_method,
                                 cl_ortho_scale=cl_ortho_scale,
                                 ortho_samples_per_task=ortho_samples_per_task,
                                 cl_penalty_clip=cl_penalty_clip)
                local_weight = self.clients[idx].get_local_data_weight()
                if local_weight <= 0:
                    self.write_log(f"skip client {idx} in warmup round {round} because local data weight is 0", prefix='warmup')
                    continue
                clients_params_arr.append(copy.deepcopy(self.clients[idx].run_manager.return_model_dict()))
                clients_data_w.append(local_weight)
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
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
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_server.run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            self.global_server.run_manager.net.module.load_state_dict(server_dict)
            
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
            self.global_server.run_manager.save_model(checkpoint, model_name="warmup.pth.tar")
            
        checkpoint = {}
        checkpoint['warmup_round'] = self.config.warmup_n_rounds
        checkpoint['warmup'] = False
        checkpoint['state_dict'] = self.global_server.net.state_dict()
        for id in self.clients_idx_arr:
            checkpoint[f"task_{self.task_id}_{id}_weight_optimizer"] = self.clients[id].run_manager.optimizer.state_dict()
            checkpoint[f"task_{self.task_id}_{id}_arch_optimizer"] = self.clients[id].arch_optimizer.state_dict()
        self.global_server.run_manager.save_model(checkpoint, model_name=f"warmup.pth.tar")
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
        if not getattr(self.config, "skip_warmup", False):
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
