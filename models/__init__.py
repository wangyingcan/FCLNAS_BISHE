# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from models.baseline_nets import BaselineResNet
from models.normal_nets.proxyless_nets import ProxylessNASNets


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    elif name == BaselineResNet.__name__:
        return BaselineResNet
    else:
        raise ValueError('unrecognized type of network: %s' % name)
