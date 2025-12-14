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

    
    def run(self):
        print('self.config.resume: ', self.config.resume)
        if self.config.resume:
            try:
                print('loading global_run_manager model:')
                self.global_run_manager.load_model()
                for id in self.clients_idx_arr:
                    print(id)
                    self.clients[id].load_clients_opt()
            except Exception as e:
                print('Exception about load clients opt:', e)

        self.start_round = self.global_run_manager.round
        # 如果不是断点恢复，将 round 重置为 0，避免继承旧 checkpoint 的 round 导致轮数偏移
        if not getattr(self.config, "resume", False):
            self.global_run_manager.round = 0
            self.start_round = 0
        print(f"[Retrain] start_round={self.start_round}, last_round={self.last_round}")
        best_val_acc = 0
        for round in range(self.start_round, self.last_round):
            print('round', round+1)
            clients_trn_loss, clients_trn_top1, clients_trn_top5, clients_val_loss, clients_val_top1, clients_val_top5, clients_lr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            clients_params_arr, clients_data_w = [], []
            round_time = time.time()
            server_model = copy.deepcopy(self.global_run_manager.net.module)
            start_local_epoch, last_local_epoch = arrange_local_epoch_from_round(global_round=round,
                                                                                 local_epoch_number=self.local_epoch_number)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for idx in self.clients_idx_arr:
                trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = self.clients[idx].train_run_manager(
                    start_local_epoch=start_local_epoch,
                    last_local_epoch=last_local_epoch,
                    server_model=server_model,
                    writer=self.writerTf,
                    global_round_idx=round + 1)
                clients_params_arr.append(copy.deepcopy(self.clients[idx].return_model_dict()))
                clients_data_w.append(self.clients[idx].get_local_data_weight())
                clients_trn_loss.update(trn_loss)
                clients_trn_top1.update(trn_top1)
                clients_trn_top5.update(trn_top5)
                clients_val_loss.update(val_loss)
                clients_val_top1.update(val_top1)
                clients_val_top5.update(val_top5)
                clients_lr.update(lr)
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

            # 联邦聚合
            new_weight_fedavg = average_weights(clients_params_arr, clients_data_w)
            server_dict = self.global_run_manager.net.module.state_dict()
            server_dict.update(new_weight_fedavg)
            missing, unexpected = self.global_run_manager.net.module.load_state_dict(server_dict, strict=False)
            
            # 记录多余的或缺失的键
            if missing or unexpected:
                self.write_log(
                    f"[FEDAVG] round {round} load_state_dict missing={len(missing)}, unexpected={len(unexpected)}",
                    prefix="retrain",
                    should_print=True,
                )

            # 在验证集上跑一下 + 保存最优模型
            self.global_run_manager.net.module.eval()
            with torch.no_grad():
                val_loss, val_acc_top1, val_acc_top5 = self.global_run_manager.validate(is_test=False, return_top5=True)
                if best_val_acc < val_acc_top1:
                    best_val_acc = val_acc_top1
                    is_best = True
                else:
                    is_best = False
                
                checkpoint = {}
                checkpoint['round'] = round
                checkpoint['state_dict'] = self.global_run_manager.net.module.state_dict()
                for id in self.clients_idx_arr:
                    checkpoint[str(id) + '_weight_optimizer'] = self.clients[
                        id].optimizer.state_dict()
                self.global_run_manager.save_model(checkpoint, is_best=is_best)
                
                self.writerTf.add_scalar('global_test_loss', val_loss, round)
                self.writerTf.add_scalar('global_test_top1', val_acc_top1, round)
                self.writerTf.add_scalar('global_test_top5', val_acc_top5, round)
                    
            # 在测试集上跑一下 + 正则化
            self.test_inference(tag="learned_net")
            # 在线累积 EWC：按设定轮次或最后一轮计算 Fisher
            if (
                getattr(self.config, "ewc_lambda", 0) > 0
                and getattr(self.config, "ewc_online_interval", 0) > 0
                and (
                    (round + 1) % int(self.config.ewc_online_interval) == 0
                    or round == self.last_round - 1
                )
            ):
                fisher, processed = self.global_run_manager.compute_importance(
                    max_samples=getattr(self.config, "ewc_samples_per_task", None)
                )
                if fisher is not None:
                    # 在线阶段仅累积 Fisher，不更新参考参数，避免 anchor 随当前任务漂移
                    self.global_run_manager.consolidate_ewc(fisher, update_prev_params=False)
                    stats = {
                        "fisher_keys": len(fisher),
                        "fisher_norm": sum(v.sum().item() for v in fisher.values()),
                        "processed": processed,
                    }
                    self.write_log(
                        f"[EWC] round {round} consolidate fisher_keys={stats['fisher_keys']} "
                        f"fisher_norm={stats['fisher_norm']:.4f} processed={stats['processed']}",
                        prefix="retrain",
                        should_print=True,
                    )
                else:
                    self.write_log(
                        f"[EWC] round {round} fisher is None (processed={processed}), skip consolidate",
                        prefix="retrain",
                        should_print=True,
                    )
        self.writerTf.close()

    
    def get_server(self):
        return copy.deepcopy(self.global_run_manager)

    
    def get_model(self):
        return copy.deepcopy(self.global_run_manager.net.module)

    
    def get_weights(self):
        return copy.deepcopy(self.global_run_manager.net.module.state_dict())

    
    def test_inference(self, tag=None):
        # Test inference after completion of training
        val_loss, val_acc_top1, val_acc_top5 = self.global_run_manager.validate(is_test=True, return_top5=True)
        prefix_name = tag if tag is not None else self.hardware
        self.write_log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " +
            "{},test_eval loss {:.4f}, top1 {:.4f}, top5 {:.4f}".format(
                prefix_name, val_loss, val_acc_top1, val_acc_top5 
            ),
            prefix='test',
            should_print=True
        )

    
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
