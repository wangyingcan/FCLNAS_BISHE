#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse

import os

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import copy
import pickle
import time
import warnings
from commonwealth_machine import *
import numpy as np
import torch
from utils_old import *
import nas_utils
from models.super_nets.super_proxyless import *
from models.normal_nets.proxyless_nets import *
from utils.pytorch_utils import create_exp_dir
from run_manager import CifarRunConfig

warnings.filterwarnings('ignore')

ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms

    'mobile': {
        '1.00': 80,
    },
    'cpu': {'1.00': 80, },
    'gpu8': {'1.00': 80, },
}

parser = argparse.ArgumentParser()
""" Federated Learning """
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
parser.add_argument('--object_to_retrain', type=str, default='supernet', choices=['supernet', 'cpu', 'gpu8', 'flops'],
                    help="0:supernet, 1:cpu, 2:gpu8, 3:flops.")
parser.add_argument('--iid', type=int, default=0, help='Default set 1 to IID. Set to 0 for non-IID.')
parser.add_argument('--unequal', type=int, default=1, help='whether to use unequal data splits for  \
                                non-i.i.d setting (use 0 for equal splits)')

""" ProxylessNAS """
parser.add_argument('--warmup', action='store_true', help='if have not warmup, please set it True')
parser.add_argument('--path', type=str, default='output/fednas-grad12,3,4,3,4,3/learned_net', help='checkpoint save path')
parser.add_argument('--save_env', type=str, default='EXP', help='experiment time name')
parser.add_argument('--resume', action='store_true', help='load last checkpoint')  # load last checkpoint
parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--start_round', default=0, type=int, help='start round in fed_search')
parser.add_argument('--last_round', default=15, type=int, help='last round in fed_search')
parser.add_argument('--local_epoch_number', default=5, type=int, help='local epoch each round in fed_search')

""" run config """
parser.add_argument('--client_id', type=int, default=10, help='local single client id')
parser.add_argument('--dataset_location', type=str, default='/dataset/cifar10/',
                    help='cifar dataset path. e.g. /dataset/cifar10/ ')
parser.add_argument('--n_epochs', type=int, default=1250,
                    help='local clients full epoch numbers on single client, equal to local_epoch_number * (last_round - start_round) ')  # 单个client上跑的epochs
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10'])
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--valid_size', type=int, default=50000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=0.0003)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=2)
parser.add_argument('--print_frequency', type=int, default=10)
parser.add_argument('--n_worker', type=int, default=2)  # 1 is most stable. 2 or 4 is bad. 3 is also ok.
parser.add_argument('--search', action='store_true', help='do not use it. it is retrain and not search')
parser.add_argument('--cifar10', action='store_true', help='if you use cifar100, do not use it')

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320',
                    help="width (output channels) of each cell stage in the block, also last_channel = make_divisible(400 * width_mult, 8) if width_mult > 1.0 else 400")
parser.add_argument('--n_cell_stages', type=str, default='2,3,4,3,4,3', help="number of cells in each cell stage")
parser.add_argument('--stride_stages', type=str, default='1,1,2,1,2,1', help="stride of each cell stage in the block")
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='rl', choices=['grad', 'rl'])
parser.add_argument('--warmup_n_rounds', type=int, default=40)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=0.025)  # 1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None,
                    choices=['mobile', 'cpu', 'gpu8', None, 'flops'])
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)

args = parser.parse_args()
args.n_epochs = args.local_epoch_number * (args.last_round - args.start_round)
args.save_env = 'env_dir/search-{}-{}-{}-{}'.format(args.arch_algo, args.train_batch_size, args.target_hardware,
                                                    time.strftime("%Y%m%d-%H%M%S"))
print("args.path: ", args.path)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
if __name__ == '__main__':
    
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.path, exist_ok=True)

    # build run args from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    if args.no_decay_keys == 'None':
        args.no_decay_keys = None

    clients_run_config_arr = []
    run_config = None
    for idx in range(args.num_users):
        args.client_id = idx
        run_config = CifarRunConfig(
            **args.__dict__
        )
        clients_run_config_arr.append(run_config)

    # create run_config for global client.
    args.client_id = 10
    global_run_config = CifarRunConfig(
        **args.__dict__
    )
    print('Run config:')
    for k, v in global_run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path = '%s/net.config' % args.path
    net = None
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name

        net_config = json.load(open(net_config_path, 'r'))
        net = get_net_by_name(net_config['name']).build_from_config(net_config)
    else:
        print('net_config_path is not file!')

    # build run manager
    global_run_manager = RunManager(args.path, net, global_run_config)
    global_run_manager.save_config(print_info=True)

    clients, \
    mobile_client_idx_arr, cpu_client_idx_arr, \
    gpu8_client_idx_arr, flops_client_idx_arr, \
    None_client_idx_arr, all_client_idx_arr = \
        [], [], [], [], [], [], []

    for idx in range(args.num_users):
        target_hardware_arr = ['cpu', 'gpu8', 'flops', None]
        all_client_idx_arr.append(idx)
        idx_target_hardware = set_target_hardware(idx=idx)
        if idx_target_hardware == 'mobile':
            mobile_client_idx_arr.append(idx)
        elif idx_target_hardware == 'cpu':
            cpu_client_idx_arr.append(idx)
        elif idx_target_hardware == 'gpu8':
            gpu8_client_idx_arr.append(idx)
        elif idx_target_hardware == 'flops':
            flops_client_idx_arr.append(idx)
        elif idx_target_hardware == None:
            None_client_idx_arr.append(idx)
        client = RunManager(args.path, net, clients_run_config_arr[idx])
        clients.append(client)
        print("The {} user has {} training data and {} test data.".format(idx,
            client.run_config.data_provider.trn_set_length,
            client.run_config.data_provider.tst_set_length))

    if args.object_to_retrain == 'supernet':
        all_clients = CommonwealthMachine(target_hardware='supernet',
                                          config=args,
                                          global_run_manager=global_run_manager,
                                          clients_idx_arr=all_client_idx_arr,
                                          clients=clients,
                                          start_round=args.start_round,
                                          last_round=args.last_round,
                                          path=args.path,
                                          )
        all_clients.run()

    elif args.object_to_retrain == 'cpu':
        """ 

                                       stage2: pruning sub_net

        """
        print("-----------------------------1.training for cpu--------------------------------")
        cpuClusteringMachine = CommonwealthMachine(target_hardware='super_net',
                                                   config=args,
                                                   global_run_manager=global_run_manager,
                                                   clients_idx_arr=cpu_client_idx_arr,
                                                   clients=clients,
                                                   start_round=args.start_round,
                                                   last_round=args.last_round,
                                                   path=args.path,
                                                   )
        cpuClusteringMachine.run()

    elif args.object_to_retrain == 'gpu8':
        # ----------------------------------2.training for gpu8----------------------------------------
        print("---------------------------------2.pruning for gpu8------------------------------------------")
        gpu8ClusteringMachine = CommonwealthMachine(target_hardware='gpu8',
                                                    config=args,
                                                    global_run_manager=global_run_manager,
                                                    clients_idx_arr=gpu8_client_idx_arr,
                                                    clients=clients,
                                                    start_round=args.start_round,
                                                    last_round=args.last_round,
                                                    path=args.path,
                                                    )
        gpu8ClusteringMachine.run()

    elif args.object_to_retrain == 'flops':
        # ----------------------------------3.training for flops---------------------------------------
        print("------------------------3.pruning for flops--------------------------------------")
        flopsClusteringMachine = CommonwealthMachine(target_hardware='flops',
                                                     config=args,
                                                     global_run_manager=global_run_manager,
                                                     clients_idx_arr=flops_client_idx_arr,
                                                     clients=clients,
                                                     start_round=args.start_round,
                                                     last_round=args.last_round,
                                                     path=args.path,
                                                     )
        flopsClusteringMachine.run()
