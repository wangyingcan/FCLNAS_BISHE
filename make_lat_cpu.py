#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import glob, os, warnings

from run_manager import CifarRunConfig
from nas_manager import *
from models.super_nets.super_proxyless import SuperProxylessNASNets
from utils.pytorch_utils import create_exp_dir
import torchvision.datasets.folder

warnings.filterwarnings('ignore')
# ref values
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
parser.add_argument('--warmup', action='store_true', help='if have not warmup, please set it True')
parser.add_argument('--path', type=str, default='./output/proxyless-', help='checkpoint save path')
parser.add_argument('--save_env', type=str, default='EXP', help='experiment time name')
parser.add_argument('--resume', action='store_true', help='load last checkpoint')  # load last checkpoint
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--last_epoch', default=1000, type=int)

""" run config """
parser.add_argument('--partition_dataset', type=int, default=10,
                    help="set partition of dataset. 10 denotes all. And 3,5,7,9 denote 30%, 50%, 70%, 90%.")
parser.add_argument('--dataset_location', type=str, default='/dataset/cifar10/',
                    help='cifar dataset path')
parser.add_argument('--n_epochs', type=int, default=96, help="set circle cosine lr_schedule_type")
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
parser.add_argument('--lr_schedule_param', type=int, default=None)
# lr_schedule_param
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['imagenet', 'CIFAR10'])
parser.add_argument('--train_batch_size', type=int, default=4096,
                    help="1024 for 1 GPU, 2048 for 2 GPUs, 4096 for 4 GPUs")
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=5000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)
parser.add_argument('--n_worker', type=int, default=2)

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320',
                    #                                    24,40,80,96,192,320
                    #                                    32,64,96,144,288,400
                    #                                    32,64,96,192,320,640
                    #                                    32,72,112,144,288,576
                    #                                    3,2,4,3,4,3
                    #                                    2,3,4,3,4,3
                    help="width (output channels) of each cell stage in the block, also last_channel = make_divisible(400 * width_mult, 8) if width_mult > 1.0 else 400")
parser.add_argument('--n_cell_stages', type=str, default='2,3,4,3,4,3', help="number of cells in each cell stage")
parser.add_argument('--stride_stages', type=str, default='1,1,2,1,2,1', help="stride of each cell stage in the block")
parser.add_argument('--width_mult', type=float, default=1.0, help="the scale factor of width")
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'], help="gradient-based or rl-based")
parser.add_argument('--warmup_epochs', type=int, default=75)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-4)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=0.001)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['cpu', 'gpu8', 'flops', None])
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=40)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2',
                    choices=['full_v2', 'full', 'two'])  # 选哪一个？two？此处可能有干扰，超参的原因
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='add#linear', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float,
                    default=2e-1)  # grad_reg_loss_params setted as lyken answered in zhihu
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', default=False)
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)  # 设置为原始的
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)

args = parser.parse_args()
args.save_env = 'env_dir/search-{}-{}-{}'.format(args.arch_algo, args.train_batch_size, time.strftime("%Y%m%d-%H%M%S"))
args.path = "./output/proxyless-" + args.arch_algo + str(args.manual_seed)
if args.target_hardware is not None:
    args.path = "./output/proxyless-" + args.arch_algo + str(args.manual_seed) + args.target_hardware
if args.partition_dataset != 10:
    args.path = args.path + "partition_dataset" + str(args.partition_dataset)
args.path = args.path + str(args.n_cell_stages)
# see flops-latency
args.save_env = 'env_dir/search-{}-{}-{}-{}'.format(args.arch_algo, args.train_batch_size, args.target_hardware,
                                                    time.strftime("%Y%m%d-%H%M%S"))
print("args.path: ", args.path)
print("args.save_env", args.save_env)
try:
    create_exp_dir(args.save_env, scripts_to_save=glob.glob('*.py'))
except Exception as e:
    print(e)

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = CifarRunConfig(
        **args.__dict__
    )

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv2', '3x3_MBConv3',
        '3x3_MBConv4', '3x3_MBConv5',
        '3x3_MBConv6', '5x5_MBConv3'
    ]
    super_net = SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
    )

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig

        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        from nas_manager import RLArchSearchConfig

        arch_search_config = RLArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config,
                                                   warmup=args.warmup)
    arch_search_run_manager.net.cpu()
    arch_search_run_manager.net.eval()
    arch_search_run_manager.net.build_latency_table(inference_device='cpu')
    print('cpu is done')
