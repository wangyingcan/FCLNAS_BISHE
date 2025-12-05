# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch
import torch.nn as nn
import os
import shutil
import time
import yaml


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


# noinspection PyUnresolvedReferences
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
                out_h * out_w / layer.groups
    return delta_ops


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

def create_exp_dir(path, scripts_to_save=None):
    # ensure parent directories exist
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save:
        scripts_dir = os.path.join(path, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)
        for script in scripts_to_save:
            try:
                dst_file = os.path.join(scripts_dir, os.path.basename(script))
                shutil.copyfile(script, dst_file)
            except Exception as e:
                print(f"Warning: cannot copy {script}: {e}")


def compute_cpu_latency_ms_pytorch(model, input_size, iterations=None):
    model.eval()
    model = model.cpu()
    output = None
    input = torch.randn(*input_size).cpu()

    with torch.no_grad():
        for _ in range(10):
            output = model(input)
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(iterations):
                    output = model(input)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_time = elapsed_time_ms / 1000.0
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            output = model(input)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        latency = elapsed_time_ms / iterations
    torch.cuda.empty_cache()

    return latency, output


def compute_gpu_latency_ms_pytorch(model, input_size, iterations=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    output = None
    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            output = model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(iterations):
                    output = model(input)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_time = elapsed_time_ms / 1000.0
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            output = model(input)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        latency = elapsed_time_ms / iterations
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency, output


class lat_table(object):
    """
    make latency table, and write latency to .yaml
    """

    def __init__(self, hardware='cpu'):
        self.hardware_name = hardware

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def record(self, ltype: str, _input=None, output=None, expand=None, kernel=None, stride=None,
               idskip=None, latency=None):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 1.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]

        if ltype in ('expanded_conv',):
            assert None not in (expand, kernel, stride, idskip)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
        key = '-'.join(infos)
        print({key: latency})
        with open('./' + self.hardware_name + '_trim.yaml', 'a+', encoding="utf-8") as fp:
            # self.yaml_file = fp
            yaml.dump({key: latency}, fp,
                      default_flow_style=False)  # , default_flow_style=False, encoding='utf-8', allow_unicode=True)
